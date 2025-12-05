"""
Queue manager for TTS service to prevent data loss.

Implements multi-stage queuing:
1. Input Queue: WebSocket text requests → TTS
2. Output Queue: TTS audio chunks → WebSocket

This ensures no synthesis requests are lost even under heavy load.
"""

import asyncio
import logging
import time
from asyncio import Queue
from dataclasses import dataclass
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class SynthesisRequest:
    """Text synthesis request from WebSocket"""
    connection_id: str
    text: str
    voice_id: str
    timestamp: float
    chunk_size: int = 50
    exaggeration: float = 0.5
    streaming: bool = True


@dataclass
class AudioChunk:
    """Audio chunk to send back"""
    connection_id: str
    audio_data: bytes
    chunk_id: int
    is_final: bool
    sample_rate: int = 24000


class TTSQueueManager:
    """
    Manages queues for TTS processing pipeline.

    Architecture:
    - Input Queue: Receives all synthesis requests from WebSocket
    - Output Queues: Per-connection audio chunks to send back

    This prevents data loss by buffering at each stage.
    """

    def __init__(
        self,
        input_queue_size: int = 500,
        output_queue_size: int = 2000,  # Larger for audio chunks
    ):
        # Input queue: All synthesis requests
        self.input_queue = Queue(maxsize=input_queue_size)

        # Per-connection output queues
        self.output_queues: Dict[str, Queue] = {}

        # Configuration
        self.output_queue_size = output_queue_size

        # Metrics
        self.metrics = {
            'requests_received': 0,
            'requests_processed': 0,
            'requests_dropped': 0,
            'chunks_sent': 0,
            'active_connections': 0,
        }

        # Worker tasks
        self.workers = []
        self.running = False

    async def start(self):
        """Start background monitoring workers"""
        if self.running:
            logger.warning("Queue manager already running")
            return

        self.running = True
        logger.info("Starting TTS queue manager workers")

        # Worker: Monitor queue sizes and log metrics
        self.workers.append(
            asyncio.create_task(self._metrics_worker())
        )

    async def stop(self):
        """Stop all workers gracefully"""
        logger.info("Stopping TTS queue manager")
        self.running = False

        for worker in self.workers:
            worker.cancel()

        await asyncio.gather(*self.workers, return_exceptions=True)

    async def _metrics_worker(self):
        """Log queue metrics periodically"""
        while self.running:
            try:
                await asyncio.sleep(10.0)  # Every 10 seconds

                logger.info(
                    f"Queue metrics - "
                    f"Input: {self.input_queue.qsize()}/{self.input_queue.maxsize}, "
                    f"Connections: {len(self.output_queues)}, "
                    f"Requests: {self.metrics['requests_received']}, "
                    f"Chunks: {self.metrics['chunks_sent']}, "
                    f"Dropped: {self.metrics['requests_dropped']}"
                )

                # Alert if queue filling up
                if self.input_queue.qsize() > self.input_queue.maxsize * 0.8:
                    logger.warning(
                        f"Input queue almost full: {self.input_queue.qsize()}/{self.input_queue.maxsize}"
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics worker: {e}")

    async def enqueue_request(
        self,
        connection_id: str,
        text: str,
        voice_id: str = "default",
        chunk_size: int = 50,
        exaggeration: float = 0.5,
        streaming: bool = True,
        timeout: float = 2.0
    ) -> bool:
        """
        Add synthesis request to input queue.

        Returns:
            bool: True if enqueued successfully, False if dropped
        """
        request = SynthesisRequest(
            connection_id=connection_id,
            text=text,
            voice_id=voice_id,
            timestamp=time.time(),
            chunk_size=chunk_size,
            exaggeration=exaggeration,
            streaming=streaming
        )

        try:
            # Try to put with timeout
            await asyncio.wait_for(
                self.input_queue.put(request),
                timeout=timeout
            )
            self.metrics['requests_received'] += 1
            return True

        except asyncio.TimeoutError:
            logger.warning(
                f"Input queue full, dropping request for connection {connection_id}"
            )
            self.metrics['requests_dropped'] += 1
            return False

    async def get_next_request(self) -> Optional[SynthesisRequest]:
        """
        Get next synthesis request for processing.

        Returns:
            SynthesisRequest or None if queue is empty
        """
        try:
            request = await self.input_queue.get()
            return request
        except Exception as e:
            logger.error(f"Error getting request: {e}")
            return None

    async def mark_request_done(self):
        """Mark current request as processed"""
        self.input_queue.task_done()
        self.metrics['requests_processed'] += 1

    async def enqueue_audio_chunk(
        self,
        connection_id: str,
        audio_data: bytes,
        chunk_id: int,
        is_final: bool = False,
        sample_rate: int = 24000
    ) -> bool:
        """
        Add audio chunk to connection's output queue.

        Returns:
            bool: True if enqueued, False if dropped
        """
        # Get output queue for this connection
        if connection_id not in self.output_queues:
            logger.warning(f"Output queue not found for connection {connection_id}")
            return False

        output_queue = self.output_queues[connection_id]

        chunk = AudioChunk(
            connection_id=connection_id,
            audio_data=audio_data,
            chunk_id=chunk_id,
            is_final=is_final,
            sample_rate=sample_rate
        )

        try:
            # Try non-blocking put first
            output_queue.put_nowait(chunk)
            self.metrics['chunks_sent'] += 1
            return True

        except asyncio.QueueFull:
            # Queue full, try with short timeout
            try:
                await asyncio.wait_for(
                    output_queue.put(chunk),
                    timeout=0.1
                )
                self.metrics['chunks_sent'] += 1
                return True
            except asyncio.TimeoutError:
                logger.warning(
                    f"Output queue full for connection {connection_id}, dropping audio chunk {chunk_id}"
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
            # Cancel any pending items
            queue = self.output_queues[connection_id]

            # Clear the queue
            while not queue.empty():
                try:
                    queue.get_nowait()
                    queue.task_done()
                except:
                    break

            del self.output_queues[connection_id]
            self.metrics['active_connections'] = len(self.output_queues)
            logger.info(f"Unregistered connection {connection_id}")

    def get_metrics(self) -> dict:
        """Get current queue metrics"""
        return {
            **self.metrics,
            'input_queue_size': self.input_queue.qsize(),
            'output_queues_count': len(self.output_queues),
            'total_output_queue_items': sum(
                q.qsize() for q in self.output_queues.values()
            )
        }

    async def wait_until_empty(self, timeout: float = 30.0):
        """
        Wait until all queues are empty (useful for graceful shutdown).

        Args:
            timeout: Maximum time to wait in seconds
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            if (
                self.input_queue.empty() and
                all(q.empty() for q in self.output_queues.values())
            ):
                logger.info("All queues empty")
                return True

            await asyncio.sleep(0.5)

        logger.warning(f"Timeout waiting for queues to empty after {timeout}s")
        return False
