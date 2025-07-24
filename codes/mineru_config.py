#!/usr/bin/env python3
"""
MinerU Configuration Loader
Loads and validates MinerU configuration settings for Paper2Code integration
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging


class MinerUConfig:
    """
    Configuration loader and validator for MinerU integration.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration loader.
        
        Args:
            config_path: Path to configuration file (defaults to config/mineru_config.yaml)
        """
        self.logger = logging.getLogger(__name__)
        
        # Default config path
        if config_path is None:
            # Try to find config relative to this file
            current_dir = Path(__file__).parent.parent
            config_path = current_dir / "config" / "mineru_config.yaml"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._validate_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Returns:
            Configuration dictionary
        """
        if not self.config_path.exists():
            self.logger.warning(f"Config file not found: {self.config_path}")
            return self._get_default_config()
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            self.logger.info(f"Loaded configuration from: {self.config_path}")
            return config or {}
        
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration values.
        
        Returns:
            Default configuration dictionary
        """
        return {
            "mineru": {
                "installation_path": "/media/chirurgie/hdd01/Soft/GitHub/MinerU",
                "venv_path": "/media/chirurgie/hdd01/Soft/GitHub/MinerU/.venv",
                "method": "ocr",
                "timeout": 600
            },
            "gemini": {
                "model": "gemini-2.0-flash-exp",
                "vision_enhancement": True,
                "rate_limit_delay": 0.5
            },
            "paper2code": {
                "temp_dir": "temp",
                "json_conversion": {
                    "include_metadata": True,
                    "validate_format": True
                }
            },
            "logging": {
                "level": "INFO",
                "file_logging": True
            }
        }
    
    def _validate_config(self):
        """Validate configuration values."""
        # Check MinerU installation path
        mineru_path = Path(self.get("mineru.installation_path", ""))
        if not mineru_path.exists():
            self.logger.warning(f"MinerU installation path not found: {mineru_path}")
        
        # Check venv path
        venv_path = Path(self.get("mineru.venv_path", ""))
        if not venv_path.exists():
            self.logger.warning(f"MinerU venv path not found: {venv_path}")
        
        # Validate processing method
        method = self.get("mineru.method", "ocr")
        if method not in ["ocr", "auto"]:
            self.logger.warning(f"Invalid processing method: {method}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., "mineru.installation_path")
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_mineru_settings(self) -> Dict[str, Any]:
        """Get MinerU-specific settings."""
        return self.get("mineru", {})
    
    def get_gemini_settings(self) -> Dict[str, Any]:
        """Get Gemini Vision settings."""
        return self.get("gemini", {})
    
    def get_paper2code_settings(self) -> Dict[str, Any]:
        """Get Paper2Code integration settings."""
        return self.get("paper2code", {})
    
    def get_processing_settings(self) -> Dict[str, Any]:
        """Get advanced processing settings."""
        return self.get("processing", {})
    
    def is_gemini_enabled(self) -> bool:
        """Check if Gemini Vision enhancement is enabled."""
        return (self.get("gemini.vision_enhancement", False) and 
                bool(os.environ.get("GEMINI_API_KEY")))
    
    def get_output_structure(self) -> Dict[str, str]:
        """Get output directory structure."""
        return self.get("paper2code.output_structure", {
            "mineru_output": "mineru_output",
            "images": "images",
            "tables": "tables",
            "formulas": "formulas"
        })
    
    def setup_logging(self):
        """Setup logging based on configuration."""
        log_level = getattr(logging, self.get("logging.level", "INFO").upper())
        log_format = self.get("logging.format", 
                             "[%(asctime)s] %(levelname)s - %(name)s: %(message)s")
        
        # Configure basic logging
        logging.basicConfig(level=log_level, format=log_format)
        
        # Add file handler if enabled
        if self.get("logging.file_logging", True):
            logger = logging.getLogger()
            
            # Create file handler
            log_file = self.get("logging.log_file", "mineru_processing.log")
            handler = logging.FileHandler(log_file)
            handler.setLevel(log_level)
            handler.setFormatter(logging.Formatter(log_format))
            
            logger.addHandler(handler)
    
    def validate_paths(self) -> bool:
        """
        Validate all configured paths.
        
        Returns:
            True if all paths are valid
        """
        paths_to_check = [
            ("mineru.installation_path", "MinerU installation"),
            ("mineru.venv_path", "MinerU virtual environment")
        ]
        
        all_valid = True
        
        for path_key, description in paths_to_check:
            path = Path(self.get(path_key, ""))
            if not path.exists():
                self.logger.error(f"{description} path not found: {path}")
                all_valid = False
        
        return all_valid
    
    def get_environment_overrides(self) -> Dict[str, Any]:
        """
        Get configuration overrides from environment variables.
        
        Returns:
            Dictionary of environment-based overrides
        """
        overrides = {}
        
        # Check for common environment variables
        env_mappings = {
            "MINERU_PATH": "mineru.installation_path",
            "MINERU_VENV": "mineru.venv_path",
            "MINERU_METHOD": "mineru.method",
            "GEMINI_MODEL": "gemini.model",
            "LOG_LEVEL": "logging.level"
        }
        
        for env_var, config_key in env_mappings.items():
            value = os.environ.get(env_var)
            if value:
                overrides[config_key] = value
        
        return overrides
    
    def apply_environment_overrides(self):
        """Apply environment variable overrides to configuration."""
        overrides = self.get_environment_overrides()
        
        for key, value in overrides.items():
            self._set_nested_key(self.config, key, value)
            self.logger.info(f"Applied environment override: {key} = {value}")
    
    def _set_nested_key(self, config_dict: Dict, key: str, value: Any):
        """Set nested dictionary key using dot notation."""
        keys = key.split('.')
        for k in keys[:-1]:
            config_dict = config_dict.setdefault(k, {})
        config_dict[keys[-1]] = value
    
    def save_config(self, output_path: Optional[str] = None):
        """
        Save current configuration to file.
        
        Args:
            output_path: Output file path (defaults to original config path)
        """
        output_path = output_path or self.config_path
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)
            
            self.logger.info(f"Configuration saved to: {output_path}")
        
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return f"MinerUConfig(path={self.config_path}, valid_paths={self.validate_paths()})"


# Global configuration instance
_config_instance = None


def get_config(config_path: Optional[str] = None) -> MinerUConfig:
    """
    Get global configuration instance.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        MinerUConfig instance
    """
    global _config_instance
    
    if _config_instance is None or config_path is not None:
        _config_instance = MinerUConfig(config_path)
    
    return _config_instance


def main():
    """Command line interface for configuration management."""
    import argparse
    
    parser = argparse.ArgumentParser(description="MinerU Configuration Manager")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--validate", action="store_true", help="Validate configuration")
    parser.add_argument("--show", action="store_true", help="Show current configuration")
    parser.add_argument("--apply-env", action="store_true", help="Apply environment overrides")
    
    args = parser.parse_args()
    
    try:
        config = get_config(args.config)
        
        if args.apply_env:
            config.apply_environment_overrides()
            print("✅ Applied environment overrides")
        
        if args.validate:
            is_valid = config.validate_paths()
            print(f"✅ Configuration valid: {is_valid}")
            
            if not is_valid:
                return 1
        
        if args.show:
            print("📋 Current configuration:")
            print(yaml.dump(config.config, default_flow_style=False, indent=2))
        
        if not any([args.validate, args.show, args.apply_env]):
            print(f"📁 Configuration loaded from: {config.config_path}")
            print(f"🔧 MinerU path: {config.get('mineru.installation_path')}")
            print(f"🎯 Processing method: {config.get('mineru.method')}")
            print(f"👁️ Gemini enabled: {config.is_gemini_enabled()}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())