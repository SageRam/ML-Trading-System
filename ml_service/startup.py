# startup.py - Production startup script for ML Agent

import asyncio
import logging
import os
import sys
from datetime import datetime
import signal

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml_agent_enhanced import TradingMLService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ml_agent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MLAgentRunner:
    """Production runner for ML Trading Agent"""
    
    def __init__(self):
        self.service = None
        self.running = False
        self.shutdown_event = asyncio.Event()
        
    async def initialize(self):
        """Initialize the ML service with retries"""
        max_retries = 5
        retry_delay = 10
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Initializing ML Trading Service (attempt {attempt + 1}/{max_retries})...")
                
                self.service = TradingMLService()
                await self.service.initialize()
                
                if self.service.is_initialized:
                    logger.info("✅ ML Trading Service initialized successfully!")
                    return True
                
            except Exception as e:
                logger.error(f"Failed to initialize service: {e}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
        
        return False
    
    async def health_check(self):
        """Periodic health check"""
        while self.running:
            try:
                # Check Redis connection
                if self.service.redis_client:
                    await self.service.redis_client.ping()
                
                # Check database connection
                if self.service.db_engine:
                    with self.service.db_engine.connect() as conn:
                        conn.execute("SELECT 1")
                
                logger.info("✅ Health check passed")
                
            except Exception as e:
                logger.error(f"❌ Health check failed: {e}")
                # Could implement alerting here
            
            await asyncio.sleep(300)  # Check every 5 minutes
    
    async def run(self):
        """Main run loop"""
        self.running = True
        
        # Initialize service
        if not await self.initialize():
            logger.error("Failed to initialize service after all retries")
            return
        
        # Check if models need training
        if not self.service.predictor.is_trained:
            logger.info("No trained models found. Starting initial training...")
            result = await self.service.retrain_model()
            if result.get('status') != 'success':
                logger.error("Failed to train initial models. Running in monitoring mode...")
            else:
                logger.info("✅ Initial model training completed successfully!")
        
        # Start health check task
        health_task = asyncio.create_task(self.health_check())
        
        # Configuration
        symbols = ["USDJPY", "EURUSD", "GBPUSD"]
        signal_interval = 900  # 15 minutes
        retrain_interval = 3600  # 1 hour
        
        last_retrain = datetime.now()
        
        logger.info(f"Starting main loop - Monitoring {len(symbols)} symbols")
        logger.info(f"Signal interval: {signal_interval}s, Retrain interval: {retrain_interval}s")
        
        while self.running:
            try:
                loop_start = datetime.now()
                
                # Generate signals for all symbols
                signal_count = 0
                for symbol in symbols:
                    try:
                        logger.info(f"Analyzing {symbol}...")
                        analysis = await self.service.analyze_market(symbol)
                        
                        if analysis.get('signal'):
                            signal_count += 1
                            logger.info(f"📊 Generated signal for {symbol}:")
                            logger.info(f"  Direction: {analysis['signal']['direction']}")
                            logger.info(f"  Confidence: {analysis['signal']['confidence']:.2%}")
                            logger.info(f"  Entry: {analysis['signal']['entry_price']}")
                            
                            # Send signal to API
                            try:
                                import requests
                                response = requests.post(
                                    f"{self.service.API_URL}/api/signals",
                                    json=analysis['signal'],
                                    timeout=30,
                                    headers={'Content-Type': 'application/json'}
                                )
                                
                                if response.status_code == 200:
                                    logger.info(f"✅ Signal sent successfully for {symbol}")
                                else:
                                    logger.warning(f"⚠️ API returned status {response.status_code}")
                                    
                            except Exception as e:
                                logger.error(f"❌ Failed to send signal: {e}")
                        
                    except Exception as e:
                        logger.error(f"Error analyzing {symbol}: {e}")
                
                # Log summary
                logger.info(f"📈 Loop completed: {signal_count} signals generated")
                
                # Check if retraining is needed
                if (datetime.now() - last_retrain).total_seconds() > retrain_interval:
                    logger.info("🔄 Starting scheduled model retraining...")
                    try:
                        result = await self.service.retrain_model()
                        if result.get('status') == 'success':
                            logger.info("✅ Model retraining completed successfully!")
                            logger.info(f"  Accuracy: {result.get('accuracy', 0):.2%}")
                        last_retrain = datetime.now()
                    except Exception as e:
                        logger.error(f"❌ Retraining failed: {e}")
                
                # Log performance metrics
                try:
                    metrics = await self.service.get_performance_metrics()
                    logger.info(f"📊 Performance Metrics:")
                    logger.info(f"  Models trained: {metrics['ml_model']['is_trained']}")
                    logger.info(f"  Signals (24h): {metrics['signals']['last_24h']}")
                    logger.info(f"  Signal distribution: {metrics['signals']['distribution']}")
                except Exception as e:
                    logger.error(f"Failed to get metrics: {e}")
                
                # Calculate sleep time
                loop_duration = (datetime.now() - loop_start).total_seconds()
                sleep_time = max(0, signal_interval - loop_duration)
                
                if sleep_time > 0:
                    logger.info(f"💤 Sleeping for {sleep_time:.0f} seconds...")
                    
                    # Use shutdown event for graceful shutdown
                    try:
                        await asyncio.wait_for(
                            self.shutdown_event.wait(),
                            timeout=sleep_time
                        )
                        # If we get here, shutdown was requested
                        break
                    except asyncio.TimeoutError:
                        # Normal timeout, continue loop
                        pass
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error
        
        # Cleanup
        health_task.cancel()
        await self.cleanup()
    
    async def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up resources...")
        
        if self.service:
            if self.service.redis_client:
                await self.service.redis_client.close()
            
            if self.service.db_engine:
                self.service.db_engine.dispose()
        
        logger.info("✅ Cleanup completed")
    
    def shutdown(self):
        """Signal shutdown"""
        logger.info("Shutdown requested...")
        self.running = False
        self.shutdown_event.set()

# Global runner instance
runner = None

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {signum}")
    if runner:
        runner.shutdown()

async def main():
    """Main entry point"""
    global runner
    
    logger.info("=" * 50)
    logger.info("ML Trading Agent Starting")
    logger.info(f"Time: {datetime.now()}")
    logger.info(f"PID: {os.getpid()}")
    logger.info("=" * 50)
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create and run the service
    runner = MLAgentRunner()
    
    try:
        await runner.run()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
    finally:
        logger.info("ML Trading Agent shutting down...")

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())