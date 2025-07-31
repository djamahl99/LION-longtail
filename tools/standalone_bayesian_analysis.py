"""
Bayesian Parameter Modeling for Long-tail Detection Analysis - AV2 Integration
Refactored to use AV2 detection utilities for proper assignment and evaluation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from dataclasses import dataclass

import pymc as pm
import arviz as az

# Import AV2 utilities
from av2.evaluation.detection.utils import (
    accumulate, assign, compute_affinity_matrix, distance,
    DetectionCfg, AffinityType, DistanceType
)

from standalone_analyze_longtail import EvaluationConfig, StandaloneLongTailEvaluator


@dataclass
class BayesianModelConfig:
    """Configuration for Bayesian modeling."""
    n_samples: int = 2000
    n_chains: int = 4
    n_tune: int = 1000
    random_seed: int = 42


class BayesianParameterAnalyzer:
    """
    Bayesian analysis of detection parameters using AV2 evaluation results.
    
    This version integrates with AV2's detection utilities to leverage their
    sophisticated assignment and distance computation methods.
    """
    
    def __init__(self, config: BayesianModelConfig = None):
        self.config = config or BayesianModelConfig()
        
    def extract_av2_assignments(self, 
                               eval_dts: pd.DataFrame, 
                               eval_gts: pd.DataFrame,
                               cfg: DetectionCfg) -> Dict[str, Any]:
        """
        Extract detailed assignment information from AV2 evaluation results.
        
        Args:
            eval_dts: Evaluated detections DataFrame from AV2
            eval_gts: Evaluated ground truth DataFrame from AV2  
            cfg: Detection configuration used in AV2 evaluation
            
        Returns:
            Dictionary containing assignment details, TP/FP classifications, and metrics
        """
        results = {
            'assignments_by_category': {},
            'global_stats': {},
            'distance_metrics': {}
        }
        
        # Use the primary threshold (usually 2.0m) for detailed analysis
        primary_threshold_idx = len(cfg.affinity_thresholds_m) // 2
        primary_threshold = cfg.affinity_thresholds_m[primary_threshold_idx]
        threshold_col = str(primary_threshold)
        
        # Global TP/FP statistics
        all_evaluated_dts = eval_dts[eval_dts['is_evaluated']]
        tp_mask = all_evaluated_dts[threshold_col] == 1
        fp_mask = all_evaluated_dts[threshold_col] == 0
        
        results['global_stats'] = {
            'tp_scores': all_evaluated_dts[tp_mask]['score'].tolist(),
            'fp_scores': all_evaluated_dts[fp_mask]['score'].tolist(),
            'tp_distances': all_evaluated_dts[tp_mask]['translation_error'].tolist(),
            'tp_scale_errors': all_evaluated_dts[tp_mask]['scale_error'].tolist(),
            'tp_orientation_errors': all_evaluated_dts[tp_mask]['orientation_error'].tolist(),
            'primary_threshold': primary_threshold
        }
        
        # Per-category analysis
        for category in cfg.categories:
            cat_dts = eval_dts[(eval_dts['category'] == category) & eval_dts['is_evaluated']]
            cat_gts = eval_gts[(eval_gts['category'] == category) & eval_gts['is_evaluated']]
            
            if len(cat_dts) == 0:
                continue
                
            cat_tps = cat_dts[cat_dts[threshold_col] == 1]
            cat_fps = cat_dts[cat_dts[threshold_col] == 0]
            
            category_data = {
                'num_detections': len(cat_dts),
                'num_ground_truths': len(cat_gts),
                'num_tps': len(cat_tps),
                'num_fps': len(cat_fps),
                'tp_scores': cat_tps['score'].tolist(),
                'fp_scores': cat_fps['score'].tolist(),
                'tp_translation_errors': cat_tps['translation_error'].tolist(),
                'tp_scale_errors': cat_tps['scale_error'].tolist(),
                'tp_orientation_errors': cat_tps['orientation_error'].tolist(),
            }
            
            # Extract size information for TPs (for bias analysis)
            if len(cat_tps) > 0:
                # Get predicted sizes from detections
                pred_sizes = cat_tps[['length_m', 'width_m', 'height_m']].values
                
                # To get true sizes, we need to match back to ground truth
                # This requires re-running assignment for this category
                true_sizes = self._extract_matched_gt_sizes(cat_tps, cat_gts, cfg)
                
                if true_sizes is not None:
                    category_data['pred_sizes'] = pred_sizes
                    category_data['true_sizes'] = true_sizes
            
            results['assignments_by_category'][category] = category_data
        
        return results
    
    def _extract_matched_gt_sizes(self, 
                                 tp_detections: pd.DataFrame,
                                 category_gts: pd.DataFrame,
                                 cfg: DetectionCfg) -> Optional[np.ndarray]:
        """
        Extract the ground truth sizes that match to the given TP detections.
        
        This requires re-running assignment to get the exact matches.
        """
        if len(tp_detections) == 0 or len(category_gts) == 0:
            return None
            
        # Group by sweep for proper assignment
        matched_sizes = []
        
        for (log_id, timestamp_ns), sweep_tps in tp_detections.groupby(['log_id', 'timestamp_ns']):
            # Get corresponding ground truth for this sweep
            sweep_gts = category_gts[
                (category_gts['log_id'] == log_id) & 
                (category_gts['timestamp_ns'] == timestamp_ns)
            ]
            
            if len(sweep_gts) == 0:
                continue
                
            # Prepare data for AV2 assignment
            dts_array = sweep_tps[['tx_m', 'ty_m', 'tz_m', 'length_m', 'width_m', 
                                  'height_m', 'qw', 'qx', 'qy', 'qz']].values
            gts_array = sweep_gts[['tx_m', 'ty_m', 'tz_m', 'length_m', 'width_m', 
                                  'height_m', 'qw', 'qx', 'qy', 'qz']].values
            
            if len(dts_array) == 0 or len(gts_array) == 0:
                continue
                
            # Use AV2's assignment function
            try:
                dts_metrics, gts_metrics = assign(dts_array, gts_array, cfg)
                
                # Find which detections are TPs and get their matched GT indices
                primary_threshold_idx = len(cfg.affinity_thresholds_m) // 2
                tp_mask = dts_metrics[:, primary_threshold_idx] == 1
                
                if np.any(tp_mask):
                    # Get affinity matrix to find matches
                    affinity_matrix = compute_affinity_matrix(
                        dts_array[:, :3], gts_array[:, :3], cfg.affinity_type
                    )
                    
                    # For each TP detection, find its best matching GT
                    tp_indices = np.where(tp_mask)[0]
                    for tp_idx in tp_indices:
                        best_gt_idx = np.argmax(affinity_matrix[tp_idx])
                        # Check if this is actually a valid match
                        threshold = cfg.affinity_thresholds_m[primary_threshold_idx]
                        if affinity_matrix[tp_idx, best_gt_idx] > -threshold:
                            gt_size = gts_array[best_gt_idx, 3:6]  # length, width, height
                            matched_sizes.append(gt_size)
                            
            except Exception as e:
                print(f"Warning: Assignment failed for sweep {log_id}:{timestamp_ns}: {e}")
                continue
        
        return np.array(matched_sizes) if matched_sizes else None
    
    def model_score_distributions(self, 
                                 tp_scores: List[float], 
                                 fp_scores: List[float]) -> Dict:
        """Model score distributions for TP and FP using Beta distributions."""
        if len(tp_scores) < 10 or len(fp_scores) < 10:
            return {'error': 'Insufficient data for modeling'}
            
        with pm.Model() as score_model:
            # Prior for TP scores (expect higher)
            tp_alpha = pm.Gamma('tp_alpha', alpha=2, beta=0.5)
            tp_beta = pm.Gamma('tp_beta', alpha=2, beta=2)
            
            # Prior for FP scores (expect lower)
            fp_alpha = pm.Gamma('fp_alpha', alpha=2, beta=2)
            fp_beta = pm.Gamma('fp_beta', alpha=2, beta=0.5)
            
            # Likelihoods
            tp_obs = pm.Beta('tp_scores', alpha=tp_alpha, beta=tp_beta, 
                            observed=np.array(tp_scores))
            fp_obs = pm.Beta('fp_scores', alpha=fp_alpha, beta=fp_beta,
                            observed=np.array(fp_scores))
            
            # Sample
            trace = pm.sample(
                self.config.n_samples,
                tune=self.config.n_tune,
                chains=self.config.n_chains,
                random_seed=self.config.random_seed,
                return_inferencedata=True
            )
        
        # Compute optimal threshold
        tp_posterior_mean = trace.posterior['tp_alpha'].mean() / (
            trace.posterior['tp_alpha'].mean() + trace.posterior['tp_beta'].mean()
        )
        fp_posterior_mean = trace.posterior['fp_alpha'].mean() / (
            trace.posterior['fp_alpha'].mean() + trace.posterior['fp_beta'].mean()
        )
        
        optimal_threshold = (tp_posterior_mean + fp_posterior_mean) / 2
        
        return {
            'trace': trace,
            'optimal_threshold': float(optimal_threshold),
            'tp_mean': float(tp_posterior_mean),
            'fp_mean': float(fp_posterior_mean),
            'model': score_model,
            'n_tp_samples': len(tp_scores),
            'n_fp_samples': len(fp_scores)
        }
    
    def model_size_errors_av2(self, 
                             true_sizes: np.ndarray,
                             predicted_sizes: np.ndarray,
                             class_name: str) -> Dict:
        """
        Model size prediction errors using AV2 extracted data.
        
        Args:
            true_sizes: (N, 3) array of [length, width, height] from matched GTs
            predicted_sizes: (N, 3) array of [length, width, height] from TPs
            class_name: Name of the object class
        """
        if len(true_sizes) < 10 or len(predicted_sizes) < 10:
            return {'error': f'Insufficient data for {class_name}: {len(true_sizes)} samples'}
            
        errors = predicted_sizes - true_sizes
        
        with pm.Model() as size_model:
            # Hyperpriors for error distribution  
            mu_bias = pm.Normal('mu_bias', mu=0, sigma=1, shape=3)
            sigma_bias = pm.HalfNormal('sigma_bias', sigma=1, shape=3)
            
            # Error model - allows for systematic bias
            bias = pm.Normal('bias', mu=mu_bias, sigma=sigma_bias, shape=3)
            noise_sigma = pm.HalfNormal('noise_sigma', sigma=0.5, shape=3)
            
            # Likelihood
            size_errors = pm.Normal('size_errors', 
                                   mu=bias,
                                   sigma=noise_sigma,
                                   observed=errors)
            
            # Sample
            trace = pm.sample(
                self.config.n_samples,
                tune=self.config.n_tune,
                chains=self.config.n_chains,
                random_seed=self.config.random_seed,
                return_inferencedata=True
            )
        
        # Extract bias estimates
        bias_estimates = {
            'length_bias': float(trace.posterior['bias'].sel(bias_dim_0=0).mean()),
            'width_bias': float(trace.posterior['bias'].sel(bias_dim_0=1).mean()),
            'height_bias': float(trace.posterior['bias'].sel(bias_dim_0=2).mean()),
            'length_uncertainty': float(trace.posterior['noise_sigma'].sel(noise_sigma_dim_0=0).mean()),
            'width_uncertainty': float(trace.posterior['noise_sigma'].sel(noise_sigma_dim_0=1).mean()),
            'height_uncertainty': float(trace.posterior['noise_sigma'].sel(noise_sigma_dim_0=2).mean()),
            'n_samples': len(true_sizes)
        }
        
        return {
            'trace': trace,
            'bias_estimates': bias_estimates,
            'model': size_model,
            'class_name': class_name
        }
    
    def model_av2_error_relationships(self,
                                    scores: List[float],
                                    translation_errors: List[float],
                                    scale_errors: List[float],
                                    orientation_errors: List[float]) -> Dict:
        """
        Model relationships between confidence scores and AV2 error metrics.
        
        Uses the translation, scale, and orientation errors computed by AV2.
        """
        if len(scores) < 20:
            return {'error': 'Insufficient data for error relationship modeling'}
            
        # Ensure all arrays are the same length
        min_len = min(len(scores), len(translation_errors), len(scale_errors), len(orientation_errors))
        scores = np.array(scores[:min_len])
        trans_err = np.array(translation_errors[:min_len])
        scale_err = np.array(scale_errors[:min_len])
        orient_err = np.array(orientation_errors[:min_len])
        
        with pm.Model() as error_model:
            # Standardize predictors
            scores_std = (scores - np.mean(scores)) / np.std(scores)
            
            # Priors for each error type
            # Translation error model
            trans_intercept = pm.Normal('trans_intercept', mu=1.0, sigma=1)
            trans_beta = pm.Normal('trans_beta_score', mu=-0.5, sigma=1)
            trans_sigma = pm.HalfNormal('trans_sigma', sigma=1)
            
            trans_mu = trans_intercept + trans_beta * scores_std
            trans_obs = pm.Normal('translation_errors', mu=trans_mu, sigma=trans_sigma,
                                 observed=trans_err)
            
            # Scale error model (between 0 and 1)
            scale_intercept = pm.Normal('scale_intercept', mu=0, sigma=1)
            scale_beta = pm.Normal('scale_beta_score', mu=-0.5, sigma=1)
            scale_phi = pm.Gamma('scale_phi', alpha=2, beta=0.5)
            
            scale_eta = scale_intercept + scale_beta * scores_std
            scale_mu = pm.math.sigmoid(scale_eta)
            
            # Beta likelihood for scale errors (in [0,1])
            scale_alpha = scale_mu * scale_phi
            scale_beta_param = (1 - scale_mu) * scale_phi
            scale_obs = pm.Beta('scale_errors', alpha=scale_alpha, beta=scale_beta_param,
                               observed=np.clip(scale_err, 0.001, 0.999))
            
            # Orientation error model
            orient_intercept = pm.Normal('orient_intercept', mu=1.0, sigma=1)
            orient_beta = pm.Normal('orient_beta_score', mu=-0.3, sigma=1)
            orient_sigma = pm.HalfNormal('orient_sigma', sigma=1)
            
            orient_mu = orient_intercept + orient_beta * scores_std
            orient_obs = pm.Normal('orientation_errors', mu=orient_mu, sigma=orient_sigma,
                                  observed=orient_err)
            
            # Sample
            trace = pm.sample(
                self.config.n_samples,
                tune=self.config.n_tune,
                chains=self.config.n_chains,
                random_seed=self.config.random_seed,
                return_inferencedata=True
            )
        
        # Extract effect sizes
        effects = {
            'translation_score_effect': float(trace.posterior['trans_beta_score'].mean()),
            'scale_score_effect': float(trace.posterior['scale_beta_score'].mean()),
            'orientation_score_effect': float(trace.posterior['orient_beta_score'].mean()),
            'score_std': float(np.std(scores)),
            'n_samples': min_len
        }
        
        return {
            'trace': trace,
            'effects': effects,
            'model': error_model
        }
    
    def create_av2_diagnostic_plots(self, model_results: Dict, output_path: str):
        """Create diagnostic plots for AV2-integrated Bayesian models."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Score distributions
        if 'score_model' in model_results and 'error' not in model_results['score_model']:
            score_res = model_results['score_model']
            
            ax = axes[0, 0]
            scores = np.linspace(0, 1, 100)
            
            # Sample from posteriors and plot
            trace = score_res['trace']
            tp_alpha_samples = trace.posterior['tp_alpha'].values.flatten()[-50:]
            tp_beta_samples = trace.posterior['tp_beta'].values.flatten()[-50:]
            fp_alpha_samples = trace.posterior['fp_alpha'].values.flatten()[-50:]
            fp_beta_samples = trace.posterior['fp_beta'].values.flatten()[-50:]
            
            for i in range(min(20, len(tp_alpha_samples))):
                tp_y = stats.beta.pdf(scores, tp_alpha_samples[i], tp_beta_samples[i])
                fp_y = stats.beta.pdf(scores, fp_alpha_samples[i], fp_beta_samples[i])
                ax.plot(scores, tp_y, 'b-', alpha=0.1)
                ax.plot(scores, fp_y, 'r-', alpha=0.1)
            
            ax.axvline(score_res['optimal_threshold'], color='green', linestyle='--', 
                      label=f'Optimal: {score_res["optimal_threshold"]:.3f}')
            ax.set_xlabel('Score')
            ax.set_ylabel('Density')
            ax.set_title(f'Score Distributions\n(TP: {score_res["n_tp_samples"]}, FP: {score_res["n_fp_samples"]})')
            ax.legend(['TP', 'FP', 'Optimal'])
        
        # 2. Size bias estimates
        if 'size_models' in model_results:
            ax = axes[0, 1]
            
            classes = []
            length_biases = []
            width_biases = []
            height_biases = []
            
            for class_name, results in model_results['size_models'].items():
                if 'error' in results:
                    continue
                bias_est = results['bias_estimates']
                classes.append(f"{class_name}\n(n={bias_est['n_samples']})")
                length_biases.append(bias_est['length_bias'])
                width_biases.append(bias_est['width_bias'])
                height_biases.append(bias_est['height_bias'])
            
            if classes:
                x = np.arange(len(classes))
                width = 0.25
                ax.bar(x - width, length_biases, width, label='Length', alpha=0.8)
                ax.bar(x, width_biases, width, label='Width', alpha=0.8)
                ax.bar(x + width, height_biases, width, label='Height', alpha=0.8)
                
                ax.set_xticks(x)
                ax.set_xticklabels(classes, rotation=45, ha='right')
                ax.set_ylabel('Bias (meters)')
                ax.set_title('Size Prediction Biases by Class')
                ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
                ax.legend()
        
        # 3. Error relationships
        if 'error_model' in model_results and 'error' not in model_results['error_model']:
            effects = model_results['error_model']['effects']
            
            ax = axes[0, 2]
            effect_names = ['Translation', 'Scale', 'Orientation']
            effect_values = [
                effects['translation_score_effect'],
                effects['scale_score_effect'], 
                effects['orientation_score_effect']
            ]
            
            bars = ax.bar(effect_names, effect_values, 
                         color=['blue', 'orange', 'green'], alpha=0.7)
            ax.set_ylabel('Score Effect (Standardized)')
            ax.set_title(f'Score Effects on AV2 Errors\n(n={effects["n_samples"]})')
            ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
            
            # Add value labels on bars
            for bar, val in zip(bars, effect_values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01 if height >= 0 else height - 0.01,
                       f'{val:.3f}', ha='center', va='bottom' if height >= 0 else 'top')
        
        # 4-6. Additional diagnostic plots for model quality
        # Trace plots, posterior distributions, etc.
        for i, (model_name, ax) in enumerate(zip(['score_model', 'size_models', 'error_model'], 
                                                axes[1, :])):
            if model_name in model_results:
                if model_name == 'size_models':
                    # Show trace for first available size model
                    for class_name, size_model in model_results[model_name].items():
                        if 'error' not in size_model:
                            az.plot_trace(size_model['trace'], var_names=['bias'], ax=ax)
                            ax.set_title(f'Size Bias Traces - {class_name}')
                            break
                elif 'error' not in model_results[model_name]:
                    if model_name == 'score_model':
                        az.plot_posterior(model_results[model_name]['trace'], 
                                        var_names=['tp_alpha', 'tp_beta'], ax=ax)
                        ax.set_title('Score Model Posteriors')
                    elif model_name == 'error_model':
                        az.plot_posterior(model_results[model_name]['trace'], 
                                        var_names=['trans_beta_score', 'scale_beta_score'], ax=ax)
                        ax.set_title('Error Model Posteriors')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def generate_av2_bayesian_report(self, all_results: Dict) -> str:
        """Generate comprehensive Bayesian analysis report using AV2 integration."""
        report = "# Bayesian Parameter Analysis (AV2 Integration)\n\n"
        
        # Score modeling results
        if 'score_model' in all_results and 'error' not in all_results['score_model']:
            score_results = all_results['score_model']
            report += "## Detection Score Analysis\n\n"
            report += f"- **Optimal threshold**: {score_results['optimal_threshold']:.3f}\n"
            report += f"- **Mean TP score**: {score_results['tp_mean']:.3f}\n"
            report += f"- **Mean FP score**: {score_results['fp_mean']:.3f}\n"
            report += f"- **Score separation**: {abs(score_results['tp_mean'] - score_results['fp_mean']):.3f}\n"
            report += f"- **Sample sizes**: {score_results['n_tp_samples']} TPs, {score_results['n_fp_samples']} FPs\n\n"
        
        # Size bias results
        if 'size_models' in all_results:
            report += "## Size Prediction Biases by Class\n\n"
            for class_name, results in all_results['size_models'].items():
                if 'error' in results:
                    report += f"### {class_name}: {results['error']}\n\n"
                    continue
                    
                bias = results['bias_estimates']
                report += f"### {class_name} (n={bias['n_samples']})\n"
                report += f"- **Length**: {bias['length_bias']:.3f} ± {bias['length_uncertainty']:.3f} m\n"
                report += f"- **Width**: {bias['width_bias']:.3f} ± {bias['width_uncertainty']:.3f} m\n"
                report += f"- **Height**: {bias['height_bias']:.3f} ± {bias['height_uncertainty']:.3f} m\n\n"
                
                # Interpretation
                if abs(bias['length_bias']) > 0.1:
                    direction = "overestimating" if bias['length_bias'] > 0 else "underestimating"
                    report += f"  *Model is systematically {direction} length by {abs(bias['length_bias']):.2f}m*\n"
                if abs(bias['width_bias']) > 0.1:
                    direction = "overestimating" if bias['width_bias'] > 0 else "underestimating"
                    report += f"  *Model is systematically {direction} width by {abs(bias['width_bias']):.2f}m*\n"
                if abs(bias['height_bias']) > 0.1:
                    direction = "overestimating" if bias['height_bias'] > 0 else "underestimating"
                    report += f"  *Model is systematically {direction} height by {abs(bias['height_bias']):.2f}m*\n"
                report += "\n"
        
        # AV2 error relationships
        if 'error_model' in all_results and 'error' not in all_results['error_model']:
            effects = all_results['error_model']['effects']
            report += "## AV2 Error Analysis\n\n"
            report += f"Score effects on AV2 metrics (n={effects['n_samples']}):\n\n"
            report += f"- **Translation Error**: {effects['translation_score_effect']:.3f}\n"
            report += f"- **Scale Error**: {effects['scale_score_effect']:.3f}\n"
            report += f"- **Orientation Error**: {effects['orientation_score_effect']:.3f}\n\n"
            
            # Interpretations
            if effects['translation_score_effect'] < -0.2:
                report += "*Higher confidence scores strongly associated with lower translation errors.*\n"
            if effects['scale_score_effect'] < -0.2:
                report += "*Higher confidence scores strongly associated with better scale accuracy.*\n"
            if effects['orientation_score_effect'] < -0.2:
                report += "*Higher confidence scores strongly associated with better orientation accuracy.*\n"
            report += "\n"
        
        # Summary and recommendations
        report += "## Summary and Recommendations\n\n"
        
        if 'score_model' in all_results and 'error' not in all_results['score_model']:
            sep = abs(all_results['score_model']['tp_mean'] - all_results['score_model']['fp_mean'])
            if sep > 0.3:
                report += "✅ **Good score separation** - confidence scores are discriminative.\n"
            else:
                report += "⚠️ **Poor score separation** - consider score calibration.\n"
        
        if 'size_models' in all_results:
            significant_biases = []
            for class_name, results in all_results['size_models'].items():
                if 'error' not in results:
                    bias = results['bias_estimates']
                    if any(abs(bias[f'{dim}_bias']) > 0.15 for dim in ['length', 'width', 'height']):
                        significant_biases.append(class_name)
            
            if significant_biases:
                report += f"⚠️ **Significant size biases detected** in: {', '.join(significant_biases)}\n"
                report += "   Consider post-processing calibration or retraining with balanced data.\n"
            else:
                report += "✅ **Size predictions appear well-calibrated** across classes.\n"
        
        return report


def add_bayesian_analysis_av2(evaluator, av2_results: Dict) -> Dict:
    """
    Add Bayesian analysis using AV2 evaluation results.
    
    Args:
        evaluator: The main evaluator instance
        av2_results: Results from run_av2_evaluation containing eval_dts, eval_gts, cfg
    
    Returns:
        Dictionary of Bayesian modeling results
    """
    print("Starting AV2-integrated Bayesian analysis...")
    
    bayesian = BayesianParameterAnalyzer()
    eval_dts = av2_results['eval_dts']
    eval_gts = av2_results['eval_gts']
    cfg = av2_results['cfg']
    
    # Extract detailed assignment information
    assignment_data = bayesian.extract_av2_assignments(eval_dts, eval_gts, cfg)
    
    bayesian_results = {}
    
    # 1. Model global score distributions
    global_stats = assignment_data['global_stats']
    if len(global_stats['tp_scores']) > 20 and len(global_stats['fp_scores']) > 20:
        print(f"Modeling score distributions ({len(global_stats['tp_scores'])} TPs, {len(global_stats['fp_scores'])} FPs)")
        score_results = bayesian.model_score_distributions(
            global_stats['tp_scores'], 
            global_stats['fp_scores']
        )
        bayesian_results['score_model'] = score_results
    
    # 2. Model size errors per category  
    bayesian_results['size_models'] = {}
    for category, cat_data in assignment_data['assignments_by_category'].items():
        if 'pred_sizes' in cat_data and 'true_sizes' in cat_data:
            if len(cat_data['true_sizes']) > 10:
                print(f"Modeling size errors for {category} ({len(cat_data['true_sizes'])} samples)")
                size_results = bayesian.model_size_errors_av2(
                    cat_data['true_sizes'],
                    cat_data['pred_sizes'], 
                    category
                )
                bayesian_results['size_models'][category] = size_results
    
    # 3. Model AV2 error relationships
    if (len(global_stats['tp_scores']) > 20 and 
        len(global_stats['tp_distances']) > 20):
        print("Modeling AV2 error relationships")
        error_results = bayesian.model_av2_error_relationships(
            global_stats['tp_scores'],
            global_stats['tp_distances'],
            global_stats['tp_scale_errors'],
            global_stats['tp_orientation_errors']
        )
        bayesian_results['error_model'] = error_results
    
    # Generate outputs
    if bayesian_results:
        # Create diagnostic plots
        output_path = evaluator.output_dir / "bayesian_av2_diagnostics.png"
        bayesian.create_av2_diagnostic_plots(bayesian_results, str(output_path))
        
        # Generate report
        report = bayesian.generate_av2_bayesian_report(bayesian_results)
        report_path = evaluator.output_dir / "bayesian_av2_analysis.md"
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"Bayesian analysis complete! Results saved to {evaluator.output_dir}")
    
    return bayesian_results


# Example usage with main pipeline
if __name__ == "__main__":
    # This would be integrated into the main analysis
    from pathlib import Path

    config = EvaluationConfig(
        # predictions_path="/scratch/project_mnt/S0202/uqdetche/lidar-longtail-mining/lion/output/lion_models/lion_mamba_1f_1x_argo_128dim_sparse_v2/default/eval/epoch_2/val/default/result.pkl",
        predictions_path="/scratch/project_mnt/S0202/uqdetche/lidar-longtail-mining/lion/output/lion_models/lion_mamba_1f_1x_argo_128dim_sparse_v2/default/eval/epoch_2/val/default/processed_results.feather",
        ground_truth_path="/scratch/project_mnt/S0202/uqdetche/lidar-longtail-mining/lion/data/argo2/val_anno.feather",
        dataset_dir="/scratch/project_mnt/S0202/uqdetche/lidar-longtail-mining/lion/data/argo2/sensor/val",
        output_dir="./longtail_analysis_bayesian_results"
    )
    
    evaluator = StandaloneLongTailEvaluator(config)
    
    # Run main analysis
    av2_results = evaluator.run_av2_eval()
    
    # Add Bayesian analysis
    bayesian_results = add_bayesian_analysis_av2(evaluator, av2_results)
    
    print("Bayesian analysis complete!")