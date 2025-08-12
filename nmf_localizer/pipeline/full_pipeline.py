"""
Complete NMF localization pipeline.
Integrates all components for end-to-end experiments.
"""

import torch
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import logging
import time
from datetime import datetime

from ..config.defaults import NMFConfig, DataPack
from ..core.data_processor import DataProcessor
from ..core.usm_trainer import USMTrainer
from ..core.transfer_functions import TransferFunctionProcessor
from ..core.localizer import NMFSoundLocalizer
from ..core.evaluator import Evaluator

logger = logging.getLogger(__name__)


class NMFLocalizationPipeline:
    """
    Complete pipeline for NMF-based sound source localization.
    
    This class provides a high-level interface for running complete
    localization experiments, from data processing to evaluation.
    """
    
    def __init__(self, config: Optional[NMFConfig] = None):
        """
        Initialize pipeline with configuration.
        
        Args:
            config: NMF configuration (uses defaults if None)
        """
        self.config = config or NMFConfig()
        
        # Initialize components
        self.data_processor = DataProcessor(self.config)
        self.usm_trainer = USMTrainer(self.config)
        self.tf_processor = TransferFunctionProcessor(self.config)
        self.localizer = NMFSoundLocalizer(self.config)
        self.evaluator = Evaluator(self.config)
        
        # State tracking
        self.data_pack = None
        self.usm_model = None
        self.trained_localizer = None
        
        logger.info(f"NMF Localization Pipeline initialized")
        logger.info(f"Device: {self.config.device}")
        logger.info(f"Beta: {self.config.beta}")
        
    def prepare_data(
        self, 
        data_root: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        force_reprocess: bool = False
    ) -> DataPack:
        """
        Prepare complete dataset for NMF localization.
        
        Args:
            data_root: Root directory containing angle folders
            output_dir: Optional output directory to save processed data
            force_reprocess: Whether to force reprocessing even if data exists
            
        Returns:
            Processed data pack
        """
        data_root = Path(data_root)
        
        if not data_root.exists():
            raise FileNotFoundError(f"Data root not found: {data_root}")
        
        # Check if already processed
        if output_dir:
            output_path = Path(output_dir)
            data_pack_path = output_path / "data_pack.pth"
            
            if data_pack_path.exists() and not force_reprocess:
                logger.info(f"Loading existing data pack from: {data_pack_path}")
                self.data_pack = DataPack.load(str(data_pack_path))
                return self.data_pack
        
        logger.info(f"Processing dataset from: {data_root}")
        start_time = time.time()
        
        # Process complete dataset
        self.data_pack = self.data_processor.process_full_dataset(
            str(data_root), output_dir=output_dir
        )
        
        processing_time = time.time() - start_time
        logger.info(f"Data preparation complete in {processing_time:.1f}s")
        logger.info(f"Data pack: {self.data_pack}")
        
        return self.data_pack
    
    def train_usm(
        self,
        data_pack: Optional[DataPack] = None,
        test_multiple_betas: bool = False,
        save_path: Optional[str] = None
    ) -> torch.Tensor:
        """
        Train Universal Speech Model.
        
        Args:
            data_pack: Data pack (uses self.data_pack if None)
            test_multiple_betas: Whether to test multiple beta values
            save_path: Optional path to save trained model
            
        Returns:
            Trained USM dictionary W
        """
        data_pack = data_pack or self.data_pack
        if data_pack is None or not data_pack.speaker_data:
            raise ValueError("No speaker data available for USM training")
        
        logger.info("=== USM Training ===")
        logger.info(f"Speakers: {len(data_pack.speaker_data)}")
        logger.info(f"Beta divergence: {self.config.beta}")
        
        start_time = time.time()
        
        # Move speaker data to device
        speaker_data_device = [data.to(self.config.device) for data in data_pack.speaker_data]
        
        if test_multiple_betas:
            logger.info("Testing multiple beta values...")
            W, best_beta, beta_results = self.usm_trainer.test_multiple_betas(
                speaker_data_device,
                beta_values=[0.5, 1.0, 1.5, 2.0]
            )
            logger.info(f"Best beta: {best_beta}")
            
            # Store beta test results in metadata
            if not hasattr(data_pack, 'usm_metadata'):
                data_pack.metadata['usm_metadata'] = {}
            data_pack.metadata['usm_metadata']['beta_test_results'] = beta_results
            data_pack.metadata['usm_metadata']['best_beta'] = best_beta
        else:
            W, train_info = self.usm_trainer.train_usm(speaker_data_device)
            data_pack.metadata['usm_training_info'] = train_info
        
        self.usm_model = W
        training_time = time.time() - start_time
        
        logger.info(f"USM training complete in {training_time:.1f}s")
        logger.info(f"USM dictionary shape: {W.shape}")
        
        # Save model if requested
        if save_path:
            self.usm_trainer.save_model(save_path)
            logger.info(f"USM saved to: {save_path}")
        
        return W
    
    def load_transfer_functions(
        self,
        tf_path: Optional[Union[str, Path]] = None,
        data_pack: Optional[DataPack] = None
    ) -> torch.Tensor:
        """
        Load and process transfer functions.
        
        Args:
            tf_path: Path to transfer function file (uses data_pack if None)
            data_pack: Data pack containing transfer functions
            
        Returns:
            Processed transfer functions
        """
        data_pack = data_pack or self.data_pack
        
        if tf_path:
            # Load from file
            logger.info(f"Loading transfer functions from: {tf_path}")
            H, angles, metadata = self.tf_processor.load_transfer_functions(tf_path)
        elif data_pack and data_pack.transfer_functions is not None:
            # Use from data pack
            logger.info("Using transfer functions from data pack")
            H = data_pack.transfer_functions
            angles = data_pack.angles
            metadata = data_pack.metadata
        else:
            raise ValueError("No transfer functions available")
        
        # Process transfer functions
        if data_pack and data_pack.freqs is not None:
            logger.info("Applying transfer function processing...")
            H_processed, processing_info = self.tf_processor.process_transfer_functions(
                H, angles, freqs=data_pack.freqs.numpy() if torch.is_tensor(data_pack.freqs) else data_pack.freqs
            )
            
            # Update metadata
            metadata.update(processing_info)
        else:
            H_processed = H
            logger.info("No frequency information available, using raw transfer functions")
        
        # Store processed transfer functions
        if data_pack:
            data_pack.transfer_functions = H_processed
            data_pack.angles = angles
            data_pack.metadata.update(metadata)
        
        logger.info(f"Transfer functions loaded: {H_processed.shape}")
        
        return H_processed
    
    def train_localizer(
        self,
        usm_model: Optional[torch.Tensor] = None,
        transfer_functions: Optional[torch.Tensor] = None,
        angles: Optional[torch.Tensor] = None,
        data_pack: Optional[DataPack] = None
    ) -> NMFSoundLocalizer:
        """
        Train NMF localizer with USM and transfer functions.
        
        Args:
            usm_model: USM dictionary (uses self.usm_model if None)
            transfer_functions: Transfer functions (uses data_pack if None)
            angles: Angle array (uses data_pack if None)
            data_pack: Data pack for fallback values
            
        Returns:
            Trained localizer
        """
        data_pack = data_pack or self.data_pack
        
        # Get USM model
        W = usm_model or self.usm_model
        if W is None:
            raise ValueError("No USM model available. Train USM first.")
        
        # Get transfer functions
        if transfer_functions is not None:
            H = transfer_functions
            angle_array = angles
        elif data_pack and data_pack.transfer_functions is not None:
            H = data_pack.transfer_functions
            angle_array = data_pack.angles
        else:
            raise ValueError("No transfer functions available")
        
        logger.info("=== Localizer Training ===")
        logger.info(f"USM dictionary: {W.shape}")
        logger.info(f"Transfer functions: {H.shape}")
        logger.info(f"Angles: {angle_array.shape if angle_array is not None else 'None'}")
        
        # Load components into localizer
        self.localizer.load_source_dictionary(W)
        self.localizer.load_transfer_functions(H, angle_array)

        # If processing computed frequency weights, apply them now
        try:
            if self.data_pack and 'frequency_weights' in self.data_pack.metadata:
                fw = self.data_pack.metadata['frequency_weights']
                fw_t = fw if torch.is_tensor(fw) else torch.as_tensor(
                    fw, dtype=torch.float32, device=self.config.device
                )
                self.localizer.set_frequency_weights(fw_t)
                logger.info("Applied auto-computed frequency weights to localizer")
        except Exception as e:
            logger.warning(f"Failed to set frequency weights: {e}")
        
        self.trained_localizer = self.localizer
        
        logger.info("Localizer training complete")
        logger.info(f"Mixing matrix A: {self.localizer.A.shape}")
        
        return self.localizer
    
    def evaluate(
        self,
        test_data: Optional[List[Dict]] = None,
        localizer: Optional[NMFSoundLocalizer] = None,
        n_sources: int = 1,
        save_results: bool = True,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate localization performance.
        
        Args:
            test_data: Test data (uses data_pack if None)
            localizer: Trained localizer (uses self.trained_localizer if None)
            n_sources: Number of sources to localize
            save_results: Whether to save evaluation results
            output_dir: Output directory for results
            
        Returns:
            Evaluation results
        """
        # Get test data
        if test_data is None:
            if self.data_pack and self.data_pack.test_data:
                test_data = self.data_pack.test_data
            else:
                raise ValueError("No test data available")
        
        # Get localizer
        localizer = localizer or self.trained_localizer
        if localizer is None:
            raise ValueError("No trained localizer available")
        
        logger.info("=== Evaluation ===")
        logger.info(f"Test examples: {len(test_data)}")
        logger.info(f"Target sources: {n_sources}")
        
        # Evaluate performance
        results = self.evaluator.evaluate_localization(
            localizer, test_data, n_sources=n_sources
        )
        
        # Save results if requested
        if save_results and output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save detailed results
            self.evaluator.save_evaluation_results(
                results, 
                str(output_path / "evaluation_results.pth")
            )
            
            # Save human-readable report
            report = self.evaluator.create_evaluation_report(results)
            with open(output_path / "evaluation_report.txt", 'w') as f:
                f.write(report)
            
            logger.info(f"Results saved to: {output_path}")
        
        return results
    
    def run_full_experiment(
        self,
        data_root: Union[str, Path],
        output_dir: Union[str, Path],
        n_sources: int = 1,
        test_multiple_betas: bool = False,
        force_reprocess: bool = False,
        save_models: bool = True
    ) -> Dict[str, Any]:
        """
        Run complete NMF localization experiment.
        
        Args:
            data_root: Root directory containing data
            output_dir: Output directory for results
            n_sources: Number of sources to localize
            test_multiple_betas: Whether to test multiple beta values for USM
            force_reprocess: Whether to force data reprocessing
            save_models: Whether to save trained models
            
        Returns:
            Complete experiment results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("=" * 60)
        logger.info("STARTING FULL NMF LOCALIZATION EXPERIMENT")
        logger.info("=" * 60)
        logger.info(f"Data root: {data_root}")
        logger.info(f"Output directory: {output_path}")
        logger.info(f"Configuration: {self.config.to_dict()}")
        
        experiment_start = time.time()
        experiment_results = {
            'config': self.config.to_dict(),
            'timestamp': datetime.now().isoformat(),
            'stages': {}
        }
        
        try:
            # Stage 1: Data preparation
            logger.info("\n--- STAGE 1: DATA PREPARATION ---")
            stage_start = time.time()
            
            self.prepare_data(
                data_root, 
                output_dir=output_path / "processed_data",
                force_reprocess=force_reprocess
            )
            
            experiment_results['stages']['data_preparation'] = {
                'duration': time.time() - stage_start,
                'data_pack_info': str(self.data_pack)
            }
            
            # Stage 2: USM training
            logger.info("\n--- STAGE 2: USM TRAINING ---")
            stage_start = time.time()
            
            usm_path = output_path / "models" / "usm.pth" if save_models else None
            if usm_path:
                usm_path.parent.mkdir(parents=True, exist_ok=True)
            
            self.train_usm(
                test_multiple_betas=test_multiple_betas,
                save_path=str(usm_path) if usm_path else None
            )
            
            experiment_results['stages']['usm_training'] = {
                'duration': time.time() - stage_start,
                'model_shape': list(self.usm_model.shape),
                'test_multiple_betas': test_multiple_betas
            }
            
            # Stage 3: Transfer function processing
            logger.info("\n--- STAGE 3: TRANSFER FUNCTION PROCESSING ---")
            stage_start = time.time()
            
            self.load_transfer_functions()
            
            experiment_results['stages']['transfer_function_processing'] = {
                'duration': time.time() - stage_start,
                'tf_shape': list(self.data_pack.transfer_functions.shape),
                'n_directions': len(self.data_pack.angles)
            }
            
            # Stage 4: Localizer training
            logger.info("\n--- STAGE 4: LOCALIZER TRAINING ---")
            stage_start = time.time()
            
            self.train_localizer()
            
            # Save localizer if requested
            if save_models:
                localizer_path = output_path / "models" / "localizer.pth"
                self.localizer.save_model(str(localizer_path))
                logger.info(f"Localizer saved to: {localizer_path}")
            
            experiment_results['stages']['localizer_training'] = {
                'duration': time.time() - stage_start,
                'mixing_matrix_shape': list(self.localizer.A.shape)
            }
            
            # Stage 5: Evaluation
            logger.info("\n--- STAGE 5: EVALUATION ---")
            stage_start = time.time()
            
            evaluation_results = self.evaluate(
                n_sources=n_sources,
                save_results=True,
                output_dir=output_path / "evaluation"
            )
            
            experiment_results['stages']['evaluation'] = {
                'duration': time.time() - stage_start,
                'results': evaluation_results
            }
            
            # Complete experiment
            experiment_results['total_duration'] = time.time() - experiment_start
            experiment_results['success'] = True
            
            # Save complete experiment results
            experiment_path = output_path / "experiment_results.pth"
            torch.save(experiment_results, experiment_path)
            
            # Create experiment summary
            self._create_experiment_summary(experiment_results, output_path)
            
            logger.info("=" * 60)
            logger.info("EXPERIMENT COMPLETED SUCCESSFULLY")
            logger.info(f"Total time: {experiment_results['total_duration']:.1f}s")
            logger.info(f"Accuracy: {evaluation_results['accuracy']:.1f}%")
            logger.info(f"Results saved to: {output_path}")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            experiment_results['success'] = False
            experiment_results['error'] = str(e)
            experiment_results['total_duration'] = time.time() - experiment_start
            
            # Save partial results
            experiment_path = output_path / "experiment_results_failed.pth"
            torch.save(experiment_results, experiment_path)
            
            raise
        
        return experiment_results
    
    def _create_experiment_summary(self, results: Dict[str, Any], output_dir: Path):
        """Create human-readable experiment summary."""
        summary = []
        summary.append("NMF LOCALIZATION EXPERIMENT SUMMARY")
        summary.append("=" * 50)
        summary.append(f"Timestamp: {results['timestamp']}")
        summary.append(f"Total Duration: {results['total_duration']:.1f}s")
        summary.append(f"Success: {results['success']}")
        
        # Configuration
        summary.append("\nConfiguration:")
        config = results['config']
        summary.append(f"  Beta: {config['beta']}")
        summary.append(f"  Lambda Group: {config['lambda_group']}")
        summary.append(f"  Gamma Sparse: {config['gamma_sparse']}")
        summary.append(f"  Max Iterations: {config['max_iter']}")
        summary.append(f"  Device: {config['device']}")
        
        # Stage timings
        summary.append("\nStage Durations:")
        for stage_name, stage_info in results['stages'].items():
            summary.append(f"  {stage_name}: {stage_info['duration']:.1f}s")
        
        # Results
        if 'evaluation' in results['stages']:
            eval_results = results['stages']['evaluation']['results']
            summary.append("\nEvaluation Results:")
            summary.append(f"  Accuracy: {eval_results['accuracy']:.1f}%")
            summary.append(f"  Mean Error: {eval_results['mean_error']:.1f}°")
            summary.append(f"  Processing Time: {eval_results['mean_processing_time']*1000:.1f}ms")
        
        summary.append("\n" + "=" * 50)
        
        # Save summary
        with open(output_dir / "experiment_summary.txt", 'w') as f:
            f.write("\n".join(summary))
        
        logger.info("Experiment summary saved")
    
    def save_pipeline_state(self, path: str):
        """Save complete pipeline state."""
        state = {
            'config': self.config.to_dict(),
            'data_pack': self.data_pack,
            'usm_model': self.usm_model,
            'localizer_state': self.localizer.state_dict() if self.trained_localizer else None
        }
        torch.save(state, path)
        logger.info(f"Pipeline state saved to: {path}")
    
    def load_pipeline_state(self, path: str):
        """Load complete pipeline state."""
        state = torch.load(path, map_location=self.config.device, weights_only=False)
        
        self.config = NMFConfig.from_dict(state['config'])
        self.data_pack = state['data_pack']
        self.usm_model = state['usm_model']
        
        if state['localizer_state']:
            self.localizer.load_state_dict(state['localizer_state'])
            self.trained_localizer = self.localizer
        
        logger.info(f"Pipeline state loaded from: {path}")
