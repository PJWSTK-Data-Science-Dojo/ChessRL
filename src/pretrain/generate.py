#!/usr/bin/env python3
"""
CLI script for processing chess datasets using the Luna dataset pipeline.
Supports both config files and command line arguments for maximum flexibility.
"""

import argparse
import logging
import sys
import json
import torch
from datetime import datetime
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from typing import Dict, Any

# Add src to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from pretrain.utils.render import ChessDatasetRenderer


class DatasetMetadataManager:
    """Manages metadata creation and operations for chess datasets."""
    
    @staticmethod
    def create_metadata_file(data_dir: str, total_transitions: int, num_batches: int) -> None:
        """
        Create metadata file for the dataset.
        
        Args:
            data_dir: Directory containing batch files
            total_transitions: Total number of transitions across all batches
            num_batches: Number of batch files
        """
        metadata = {
            'total_transitions': total_transitions,
            'num_batches': num_batches,
            'avg_transitions_per_batch': total_transitions / num_batches if num_batches > 0 else 0,
            'created_at': datetime.now().isoformat()
        }
        
        metadata_file = Path(data_dir) / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    @staticmethod
    def describe_dataset(data_dir: str) -> None:
        """
        Scan all batch files and recreate metadata file.
        
        Args:
            data_dir: Directory containing batch files
        """
        data_path = Path(data_dir)
        batch_files = list(data_path.glob("batch_*.pt"))
        
        if not batch_files:
            raise ValueError(f"No batch files found in {data_dir}")
        
        total_transitions = 0
        
        print(f"Scanning {len(batch_files)} batch files...")
        for batch_file in batch_files:
            try:
                batch_data = torch.load(batch_file, map_location='cpu')
                total_transitions += len(batch_data['boards'])
            except Exception as e:
                print(f"Warning: Could not read {batch_file}: {e}")
        
        DatasetMetadataManager.create_metadata_file(
            data_dir, total_transitions, len(batch_files)
        )
        
        print(f"✅ Metadata recreated: {total_transitions:,} transitions across {len(batch_files)} batches")
    
    @staticmethod
    def get_dataset_stats(data_dir: str) -> Dict[str, Any]:
        """
        Get statistics about the dataset using metadata file.
        
        Args:
            data_dir: Directory containing batch files
        
        Returns:
            Dictionary with dataset statistics
        """
        data_path = Path(data_dir)
        
        # Try to load from metadata first
        metadata_file = data_path / "metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                # Add some computed stats for compatibility
                metadata['num_batch_files'] = metadata.get('num_batches', 0)
                
                return metadata
            except Exception as e:
                return {'error': f'Could not read metadata file: {e}'}
        
        # Fallback to basic file counting
        batch_files = list(data_path.glob("batch_*.pt"))
        
        if not batch_files:
            return {'error': 'No batch files found'}
        
        return {
            'num_batch_files': len(batch_files),
            'total_transitions': 'Unknown (no metadata file found)',
            'error': 'No metadata file found.'
        }


def setup_logging(level: str = "INFO"):
    """Setup logging configuration with enhanced formatting."""
    # Clear any existing handlers to avoid duplicates
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Create formatter with more detailed information
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create and configure console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # Set up root logger
    logging.root.setLevel(getattr(logging, level.upper()))
    logging.root.addHandler(console_handler)
    
    # Set specific logger levels to reduce noise
    logging.getLogger('polars').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)
    
    # Enable debug logging for our modules
    logging.getLogger('pretrain').setLevel(getattr(logging, level.upper()))
    
    # Log the setup
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized at {level} level")


def merge_config_and_args(config: DictConfig, args: argparse.Namespace) -> DictConfig:
    """
    Merge configuration file with command line arguments.
    Command line arguments take precedence over config file.
    """
    # Create a copy to avoid modifying the original
    merged_config = OmegaConf.create(config)
    
    # Override with command line arguments if provided
    if hasattr(args, 'output_dir') and args.output_dir:
        OmegaConf.set(merged_config, 'output.output_dir', args.output_dir)
    
    if hasattr(args, 'max_games') and args.max_games:
        OmegaConf.set(merged_config, 'input.max_games', args.max_games)
    
    if hasattr(args, 'batch_size') and args.batch_size:
        OmegaConf.set(merged_config, 'output.batch_size', args.batch_size)
    
    if hasattr(args, 'max_batches') and args.max_batches:
        OmegaConf.set(merged_config, 'output.max_batches', args.max_batches)
    
    if hasattr(args, 'chunk_size') and args.chunk_size:
        OmegaConf.set(merged_config, 'input.chunk_size', args.chunk_size)
    
    if hasattr(args, 'num_processes') and args.num_processes:
        OmegaConf.set(merged_config, 'performance.num_processes', args.num_processes)
    
    if hasattr(args, 'history_len') and args.history_len:
        OmegaConf.set(merged_config, 'game.history_len', args.history_len)
    
    if hasattr(args, 'log_level') and args.log_level:
        OmegaConf.set(merged_config, 'logging.log_level', args.log_level.upper())
    
    return merged_config


def load_config(config_path: str = None) -> DictConfig:
    """
    Load configuration from file or use defaults.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration object
    """
    if config_path and Path(config_path).exists():
        print(f"Loading configuration from: {config_path}")
        return OmegaConf.load(config_path)
    else:
        # Try default config locations
        default_configs = [
            "src/config/dataset_config.yaml",
            "config/dataset_config.yaml",
            "../config/dataset_config.yaml"
        ]
        
        for default_config in default_configs:
            if Path(default_config).exists():
                print(f"Loading default configuration from: {default_config}")
                return OmegaConf.load(default_config)
        
        # Fallback to basic configuration
        print("No configuration file found, using defaults")
        return OmegaConf.create({
            'input': {'chunk_size': 1000, 'max_games': None},
            'output': {'batch_size': 10000, 'output_dir': 'processed_data', 'max_batches': None},
            'game': {'history_len': 8},
            'performance': {'num_processes': 4, 'chunk_split_factor': 4},
            'logging': {'log_level': 'INFO', 'progress_update_freq': 1000}
        })


def cmd_process(args):
    """Process a parquet dataset."""
    # Load configuration
    config = load_config(args.config)
    
    # Merge with command line arguments
    config = merge_config_and_args(config, args)
    
    # Setup logging
    log_level = config.get('logging', {}).get('log_level', 'INFO')
    setup_logging(log_level)
    
    # Get logger for structured output
    logger = logging.getLogger(__name__)
    
    # Print configuration summary
    logger.info("="*60)
    logger.info("DATASET PROCESSING CONFIGURATION")
    logger.info("="*60)
    logger.info(f"Input file: {args.parquet_file}")
    logger.info(f"Output directory: {config.output.output_dir}")
    logger.info(f"Max games: {config.input.max_games or 'unlimited'}")
    logger.info(f"Chunk size: {config.input.chunk_size:,}")
    logger.info(f"Batch size: {config.output.batch_size:,}")
    logger.info(f"Max batches: {config.output.max_batches or 'unlimited'}")
    logger.info(f"Processes: {config.performance.num_processes}")
    logger.info(f"History length: {config.game.history_len}")
    logger.info(f"Log level: {log_level}")
    logger.info("="*60)
    
    try:
        # Create renderer
        logger.info("Creating dataset renderer...")
        renderer = ChessDatasetRenderer(config)
        logger.info("Renderer created successfully")
        
        # Process dataset
        logger.info("Starting dataset processing...")
        renderer.render_dataset(args.parquet_file)
        
        logger.info("✅ Dataset processing completed successfully!")
        
    except KeyboardInterrupt:
        logger.warning("⚠️ Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Error during processing: {e}")
        logger.exception("Full error traceback:")
        sys.exit(1)


def cmd_stats(args):
    """Show dataset statistics."""
    setup_logging(args.log_level)
    
    print(f"Getting statistics for dataset in: {args.data_dir}")
    
    try:
        stats = DatasetMetadataManager.get_dataset_stats(args.data_dir)
        
        if 'error' in stats:
            print(f"❌ Error: {stats['error']}")
            sys.exit(1)
        
        print("\n=== Dataset Statistics ===")
        for key, value in stats.items():
            if key == 'created_at':
                print(f"Created: {value}")
            elif 'transitions' in key:
                if isinstance(value, (int, float)):
                    print(f"{key.replace('_', ' ').title()}: {value:,}")
                else:
                    print(f"{key.replace('_', ' ').title()}: {value}")
            else:
                print(f"{key.replace('_', ' ').title()}: {value}")
        print("===========================")
        
    except Exception as e:
        print(f"❌ Error getting statistics: {e}")
        logging.exception("Full error traceback:")
        sys.exit(1)


def cmd_describe(args):
    """Scan dataset and recreate metadata."""
    setup_logging(args.log_level)
    
    print(f"Describing dataset in: {args.data_dir}")
    
    try:
        DatasetMetadataManager.describe_dataset(args.data_dir)
        
    except Exception as e:
        print(f"❌ Error describing dataset: {e}")
        logging.exception("Full error traceback:")
        sys.exit(1)


def cmd_config_template(args):
    """Generate a configuration template."""
    template_config = {
        'input': {
            'chunk_size': 1000,
            'max_games': None
        },
        'output': {
            'batch_size': 10000,
            'output_dir': 'processed_data',
            'max_batches': None,
            'compress_boards': True,
            'compress_actions': True,
            'compress_values': True
        },
        'game': {
            'history_len': 8,
            'validate_moves': True,
            'skip_invalid_games': True
        },
        'performance': {
            'num_processes': 4,
            'chunk_split_factor': 4,
        },
        'logging': {
            'log_level': 'INFO',
            'progress_update_freq': 1000,
            'detailed_timing': False
        }
    }
    
    output_path = args.output or "dataset_config.yaml"
    OmegaConf.save(template_config, output_path)
    print(f"✅ Configuration template saved to: {output_path}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Chess Dataset Processing Pipeline for Luna",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process with default settings
  python process_dataset.py process chess_games.parquet

  # Process with custom settings
  python process_dataset.py process chess_games.parquet \\
    --config src/config/dataset_config.yaml \\
    --output-dir my_data \\
    --max-games 100000 \\
    --num-processes 8

  # Get dataset statistics
  python process_dataset.py stats processed_data

  # Recreate metadata by scanning dataset
  python process_dataset.py describe processed_data

  # Generate config template
  python process_dataset.py config-template
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Process parquet dataset')
    process_parser.add_argument('parquet_file', help='Path to input parquet file')
    process_parser.add_argument('--config', '-c', help='Path to configuration file')
    process_parser.add_argument('--output-dir', '-o', help='Output directory for processed files')
    process_parser.add_argument('--max-games', type=int, help='Maximum number of games to process')
    process_parser.add_argument('--batch-size', type=int, help='Number of transitions per batch file')
    process_parser.add_argument('--max-batches', type=int, help='Maximum number of batches to create')
    process_parser.add_argument('--chunk-size', type=int, help='Number of games per chunk')
    process_parser.add_argument('--num-processes', type=int, help='Number of parallel processes')
    process_parser.add_argument('--history-len', type=int, help='Number of board history states')
    process_parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Logging level')
    process_parser.set_defaults(func=cmd_process)
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show dataset statistics')
    stats_parser.add_argument('data_dir', help='Directory containing batch files')
    stats_parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    stats_parser.set_defaults(func=cmd_stats)
    
    # Describe command
    describe_parser = subparsers.add_parser('describe', help='Scan dataset and recreate metadata')
    describe_parser.add_argument('data_dir', help='Directory containing batch files')
    describe_parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    describe_parser.set_defaults(func=cmd_describe)
    
    # Config template command
    config_parser = subparsers.add_parser('config-template', help='Generate configuration template')
    config_parser.add_argument('--output', '-o', help='Output file path (default: dataset_config.yaml)')
    config_parser.set_defaults(func=cmd_config_template)
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Execute command
    args.func(args)


if __name__ == "__main__":
    main()