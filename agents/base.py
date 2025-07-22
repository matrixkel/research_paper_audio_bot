"""
Base agent class providing common functionality for all agents in the system.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import asyncio
from dataclasses import dataclass

from utils.config import Config

@dataclass
class AgentResult:
    """Standard result format for agent operations"""
    success: bool
    data: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class BaseAgent(ABC):
    """
    Abstract base class for all agents in the multi-agent system.
    
    Provides common functionality including:
    - Logging setup
    - Error handling
    - Configuration access
    - Async operation support
    """
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.logger = self._setup_logger()
        self.config = Config()
        
    def _setup_logger(self) -> logging.Logger:
        """Set up logging for the agent"""
        logger = logging.getLogger(f"Agent.{self.agent_name}")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def log_info(self, message: str, **kwargs):
        """Log info message with optional metadata"""
        self.logger.info(f"{message} {kwargs if kwargs else ''}")
    
    def log_error(self, message: str, error: Exception = None, **kwargs):
        """Log error message with optional exception details"""
        error_msg = f"{message}"
        if error:
            error_msg += f" - {str(error)}"
        if kwargs:
            error_msg += f" {kwargs}"
        self.logger.error(error_msg)
    
    def log_warning(self, message: str, **kwargs):
        """Log warning message"""
        self.logger.warning(f"{message} {kwargs if kwargs else ''}")
    
    async def safe_execute(self, operation_name: str, operation_func, *args, **kwargs) -> AgentResult:
        """
        Safely execute an operation with error handling and logging.
        
        Args:
            operation_name: Name of the operation for logging
            operation_func: Function to execute
            *args: Arguments for the operation function
            **kwargs: Keyword arguments for the operation function
            
        Returns:
            AgentResult: Standardized result object
        """
        try:
            self.log_info(f"Starting {operation_name}")
            
            # Execute the operation (handle both sync and async functions)
            if asyncio.iscoroutinefunction(operation_func):
                result = await operation_func(*args, **kwargs)
            else:
                result = operation_func(*args, **kwargs)
            
            self.log_info(f"Successfully completed {operation_name}")
            return AgentResult(success=True, data=result)
            
        except Exception as e:
            self.log_error(f"Failed to execute {operation_name}", e)
            return AgentResult(
                success=False,
                error=f"{operation_name} failed: {str(e)}",
                metadata={'operation': operation_name}
            )
    
    @abstractmethod
    async def process(self, input_data: Any, **kwargs) -> AgentResult:
        """
        Main processing method that each agent must implement.
        
        Args:
            input_data: Input data for processing
            **kwargs: Additional keyword arguments
            
        Returns:
            AgentResult: Standardized result object
        """
        pass
    
    def validate_config(self, required_keys: list) -> bool:
        """
        Validate that required configuration keys are present.
        
        Args:
            required_keys: List of required configuration keys
            
        Returns:
            bool: True if all required keys are present
        """
        missing_keys = []
        for key in required_keys:
            if not hasattr(self.config, key) or not getattr(self.config, key):
                missing_keys.append(key)
        
        if missing_keys:
            self.log_error(f"Missing required configuration keys: {missing_keys}")
            return False
        
        return True
    
    async def batch_process(self, items: list, process_func, max_concurrent: int = 5) -> list:
        """
        Process a list of items concurrently with a maximum concurrency limit.
        
        Args:
            items: List of items to process
            process_func: Function to process each item
            max_concurrent: Maximum number of concurrent operations
            
        Returns:
            list: List of results
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(item):
            async with semaphore:
                if asyncio.iscoroutinefunction(process_func):
                    return await process_func(item)
                else:
                    return process_func(item)
        
        results = await asyncio.gather(
            *[process_with_semaphore(item) for item in items],
            return_exceptions=True
        )
        
        return results
