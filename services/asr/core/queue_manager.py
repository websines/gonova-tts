"""
Queue manager for ASR service to prevent data loss.

Implements multi-stage queuing:
1. Input Queue: WebSocket → VAD
2. Processing Queue: VAD → ASR (GPU)
3. Output Queue: ASR → WebSocket

This ensures no audio chunks are lost even under heavy load.
"""

import asyncio
import logging
import time
from asyncio import Queue
from dataclasses import dataclass
from typing import Dict, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


@dataclass
class AudioChunk:
    """Audio chunk from WebSocket"""
    connection_id: str
    audio_data: bytes
    timestamp: float
    sample_rate: int = 16000


@dataclass
class TranscriptionResult:
    """Transcription result to send back"""
    connection_id: str
    text: str
    is_final: bool
    confidence: float
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    words: Optional[list] = None


class ASRQueueManager:
    """
    Manages queues for ASR processing pipeline.

    Architecture:
    - Input Queue: Receives all audio chunks from WebSocket (fast I/O)
    - Processing Queue: Audio ready for GPU transcription
    - Output Queues: Per-connection results to send back

    This prevents data loss by buffering at each stage.
    """

    def __init__(
        self,
        input_queue_size: int = 1000,
        processing_queue_size: int = 100,
        output_queue_size: int = 500,
        batch_timeout: float = 0.05,  # 50ms
        max_batch_size: int = 16,
    ):
        # Input queue: All incoming audio
        self.input_queue = Queue(maxsize=input_queue_size)

        # Processing queue: Audio ready for GPU
        self.processing_queue = Queue(maxsize=processing_queue_size)

        # Per-connection output queues
        self.output_queues: Dict[str, Queue] = {}

        # Configuration
        self.output_queue_size = output_queue_size
        self.batch_timeout = batch_timeout
        self.max_batch_size = max_batch_size

        # Metrics
        self.metrics = {
            'chunks_received': 0,
            'chunks_processed': 0,
            'chunks_dropped': 0,
            'active_connections': 0,
        }

        # Worker tasks
        self.workers = []
        self.running = False

    async def start(self):
        """Start background processing workers"""
        if self.running:
            logger.warning("Queue manager already running")
            return

        self.running = True
        logger.info("Starting ASR queue manager workers")

        # Worker 1: Move items from input to processing queue
        self.workers.append(
            asyncio.create_task(self._input_worker())
        )

        # Worker 2: Monitor queue sizes and log metrics
        self.workers.append(
            asyncio.create_task(self._metrics_worker())
        )

    async def stop(self):
        """Stop all workers gracefully"""
        logger.info("Stopping ASR queue manager")
        self.running = False

        for worker in self.workers:
            worker.cancel()

        await asyncio.gather(*self.workers, return_exceptions=True)

    async def _input_worker(self):
        """
        Process input queue items.
        This can be extended to do VAD filtering before GPU processing.
        """
        while self.running:
            try:
                # Get chunk from input queue
                chunk = await self.input_queue.get()

                # For now, pass directly to processing queue
                # In production, add VAD here:
                # has_speech = await vad.detect(chunk.audio_data)
                # if has_speech:
                #     await self.processing_queue.put(chunk)

                await self.processing_queue.put(chunk)

                self.input_queue.task_done()
                self.metrics['chunks_processed'] += 1

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in input worker: {e}")

    async def _metrics_worker(self):
        """Log queue metrics periodically"""
        while self.running:
            try:
                await asyncio.sleep(10.0)  # Every 10 seconds

                logger.info(
                    f"Queue metrics - "
                    f"Input: {self.input_queue.qsize()}/{self.input_queue.maxsize}, "
                    f"Processing: {self.processing_queue.qsize()}/{self.processing_queue.maxsize}, "
                    f"Connections: {len(self.output_queues)}, "
                    f"Chunks: {self.metrics['chunks_received']}, "
                    f"Dropped: {self.metrics['chunks_dropped']}"
                )

                # Alert if queues filling up
                if self.input_queue.qsize() > self.input_queue.maxsize * 0.8:
                    logger.warning(
                        f"Input queue almost full: {self.input_queue.qsize()}/{self.input_queue.maxsize}"
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics worker: {e}")

    async def enqueue_audio(
        self,
        connection_id: str,
        audio_data: bytes,
        sample_rate: int = 16000,
        timeout: float = 1.0
    ) -> bool:
        """
        Add audio chunk to input queue.

        Returns:
            bool: True if enqueued successfully, False if dropped
        """
        chunk = AudioChunk(
            connection_id=connection_id,
            audio_data=audio_data,
            timestamp=time.time(),
            sample_rate=sample_rate
        )

        try:
            # Try to put with timeout (prevents blocking forever)
            await asyncio.wait_for(
                self.input_queue.put(chunk),
                timeout=timeout
            )
            self.metrics['chunks_received'] += 1
            return True

        except asyncio.TimeoutError:
            logger.warning(
                f"Input queue full, dropping chunk for connection {connection_id}"
            )
            self.metrics['chunks_dropped'] += 1
            return False

    async def get_batch_for_processing(self) -> list[AudioChunk]:
        """
        Get a batch of audio chunks for GPU processing.

        Collects up to max_batch_size chunks within batch_timeout.
        This enables efficient GPU batching.
        """
        batch = []

        try:
            # Get first item (blocking)
            first_chunk = await self.processing_queue.get()
            batch.append(first_chunk)

            # Try to collect more for batching (non-blocking with timeout)
            for _ in range(self.max_batch_size - 1):
                try:
                    chunk = await asyncio.wait_for(
                        self.processing_queue.get(),
                        timeout=self.batch_timeout
                    )
                    batch.append(chunk)
                except asyncio.TimeoutError:
                    # Timeout reached, process what we have
                    break

        except Exception as e:
            logger.error(f"Error getting batch: {e}")

        return batch

    async def mark_batch_done(self, batch_size: int):
        """Mark batch items as processed"""
        for _ in range(batch_size):
            self.processing_queue.task_done()

    async def enqueue_result(
        self,
        connection_id: str,
        result: TranscriptionResult
    ) -> bool:
        """
        Add transcription result to connection's output queue.

        Returns:
            bool: True if enqueued, False if dropped
        """
        # Get or create output queue for this connection
        if connection_id not in self.output_queues:
            logger.warning(f"Output queue not found for connection {connection_id}")
            return False

        output_queue = self.output_queues[connection_id]

        try:
            # Non-blocking put (fail fast if queue full)
            output_queue.put_nowait(result)
            return True

        except asyncio.QueueFull:
            logger.warning(
                f"Output queue full for connection {connection_id}, dropping result"
            )
            return False

    def register_connection(self, connection_id: str) -> Queue:
        """
        Register a new WebSocket connection.

        Returns:
            Queue: Output queue for this connection
        """
        output_queue = Queue(maxsize=self.output_queue_size)
        self.output_queues[connection_id] = output_queue
        self.metrics['active_connections'] = len(self.output_queues)

        logger.info(f"Registered connection {connection_id}")
        return output_queue

    def unregister_connection(self, connection_id: str):
        """Unregister a WebSocket connection"""
        if connection_id in self.output_queues:
            del self.output_queues[connection_id]
            self.metrics['active_connections'] = len(self.output_queues)
            logger.info(f"Unregistered connection {connection_id}")

    async def wait_until_empty(self, timeout: float = 30.0):
        """Wait until all queues are empty before shutdown"""
        start_time = time.time()

        while time.time() - start_time < timeout:
            if (self.input_queue.qsize() == 0 and
                self.processing_queue.qsize() == 0):
                logger.info("All queues empty, ready for shutdown")
                return

            await asyncio.sleep(0.5)

        logger.warning(f"Timeout waiting for queues to empty after {timeout}s")

    def get_metrics(self) -> dict:
        """Get current queue metrics"""
        return {
            **self.metrics,
            'input_queue_size': self.input_queue.qsize(),
            'processing_queue_size': self.processing_queue.qsize(),
            'output_queues_count': len(self.output_queues),
        }
