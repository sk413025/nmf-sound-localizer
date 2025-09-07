#!/usr/bin/env python3
"""
Batch Angle Testing Script
Systematically test NMF reconstruction performance across multiple angles
"""

import os
import sys
import json
import numpy as np
import subprocess
import time
from datetime import datetime
from pathlib import Path

def test_single_angle(angle, debug_script='debug_nmf_with_learned_sources.py'):
    """Test a single angle and return results"""
    print(f"\n{'='*60}")
    print(f"Testing angle: {angle}")
    print('='*60)
    
    # Create a modified version of the debug script for this angle
    angle_script = f"debug_nmf_angle_{angle}.py"
    
    # Read the original script
    with open(debug_script, 'r') as f:
        script_content = f.read()
    
    # Replace the test_angle line
    modified_content = script_content.replace(
        'test_angle = "angle_90"',
        f'test_angle = "angle_{angle}"'
    )
    
    # Write temporary script
    with open(angle_script, 'w') as f:
        f.write(modified_content)
    
    start_time = time.time()
    
    try:
        # Run the test
        result = subprocess.run([
            'python', angle_script
        ], capture_output=True, text=True, timeout=300)  # 5 minute timeout
        
        execution_time = time.time() - start_time
        
        if result.returncode == 0:
            # Find the most recent JSON file
            json_files = list(Path('.').glob('nmf_reconstruction_data_*.json'))
            if json_files:
                latest_json = max(json_files, key=lambda x: x.stat().st_mtime)
                
                # Load results
                with open(latest_json, 'r') as f:
                    data = json.load(f)
                
                # Extract key metrics
                results = {
                    'angle': angle,
                    'success': True,
                    'execution_time': execution_time,
                    'original_correlation': data['correlations']['original_method'],
                    'log_space_correlation': data['correlations']['log_space_method'], 
                    'improvement_percent': data['correlations']['improvement_percent'],
                    'data_file': str(latest_json),
                    'timestamp': data['metadata']['timestamp']
                }
                
                print(f"✅ Success - Angle {angle}")
                print(f"   Original correlation: {results['original_correlation']:.4f}")
                print(f"   Log-space correlation: {results['log_space_correlation']:.4f}")  
                print(f"   Improvement: +{results['improvement_percent']:.1f}%")
                print(f"   Execution time: {execution_time:.1f}s")
                
                return results
            else:
                print(f"❌ No JSON output found for angle {angle}")
                return {'angle': angle, 'success': False, 'error': 'No JSON output'}
        else:
            print(f"❌ Script failed for angle {angle}")
            print(f"Error: {result.stderr}")
            return {'angle': angle, 'success': False, 'error': result.stderr}
            
    except subprocess.TimeoutExpired:
        print(f"⏱️ Timeout for angle {angle}")
        return {'angle': angle, 'success': False, 'error': 'Timeout'}
    except Exception as e:
        print(f"❌ Exception for angle {angle}: {e}")
        return {'angle': angle, 'success': False, 'error': str(e)}
    finally:
        # Clean up temporary script
        if os.path.exists(angle_script):
            os.remove(angle_script)

def generate_visualizations(results, output_dir='angle_visualizations'):
    """Generate visualizations for all successful angles"""
    print(f"\n{'='*60}")
    print("Generating visualizations...")
    print('='*60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    viz_results = []
    
    for result in results:
        if result['success']:
            angle = result['angle']
            data_file = result['data_file']
            
            try:
                # Generate Y vs Y_hat curves
                curves_output = f"{output_dir}/y_yhat_curves_angle_{angle}.png"
                
                subprocess.run([
                    'python', 'visualize_y_yhat_curves.py', 
                    data_file, '--output', curves_output
                ], check=True, capture_output=True)
                
                print(f"✅ Generated curves for angle {angle}: {curves_output}")
                viz_results.append({
                    'angle': angle,
                    'curves_file': curves_output,
                    'data_file': data_file
                })
                
            except subprocess.CalledProcessError as e:
                print(f"❌ Visualization failed for angle {angle}: {e}")
    
    return viz_results

def create_summary_report(results, viz_results, output_file='angle_test_summary.json'):
    """Create a comprehensive summary report"""
    print(f"\n{'='*60}")
    print("Creating summary report...")
    print('='*60)
    
    # Calculate statistics
    successful_results = [r for r in results if r['success']]
    
    if successful_results:
        original_correlations = [r['original_correlation'] for r in successful_results]
        log_correlations = [r['log_space_correlation'] for r in successful_results]
        improvements = [r['improvement_percent'] for r in successful_results]
        
        summary = {
            'test_summary': {
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                'total_angles_requested': len(results),
                'successful_tests': len(successful_results),
                'failed_tests': len(results) - len(successful_results),
                'success_rate': len(successful_results) / len(results) * 100,
                'angle_range': f"{min(r['angle'] for r in results)}-{max(r['angle'] for r in results)} degrees"
            },
            'performance_statistics': {
                'original_method': {
                    'mean_correlation': float(np.mean(original_correlations)),
                    'std_correlation': float(np.std(original_correlations)),
                    'min_correlation': float(np.min(original_correlations)),
                    'max_correlation': float(np.max(original_correlations))
                },
                'log_space_method': {
                    'mean_correlation': float(np.mean(log_correlations)),
                    'std_correlation': float(np.std(log_correlations)),
                    'min_correlation': float(np.min(log_correlations)),
                    'max_correlation': float(np.max(log_correlations))
                },
                'improvement_analysis': {
                    'mean_improvement': float(np.mean(improvements)),
                    'std_improvement': float(np.std(improvements)),
                    'min_improvement': float(np.min(improvements)),
                    'max_improvement': float(np.max(improvements)),
                    'best_angle': successful_results[np.argmax(improvements)]['angle'],
                    'worst_angle': successful_results[np.argmin(improvements)]['angle']
                }
            },
            'detailed_results': results,
            'visualization_files': viz_results
        }
        
        # Save summary
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print key findings
        print(f"📊 Summary Statistics:")
        print(f"   Successful tests: {len(successful_results)}/{len(results)} ({len(successful_results)/len(results)*100:.1f}%)")
        print(f"   Original method: {np.mean(original_correlations):.4f} ± {np.std(original_correlations):.4f}")
        print(f"   Log-space method: {np.mean(log_correlations):.4f} ± {np.std(log_correlations):.4f}")
        print(f"   Mean improvement: +{np.mean(improvements):.1f}% ± {np.std(improvements):.1f}%")
        print(f"   Best performing angle: {summary['performance_statistics']['improvement_analysis']['best_angle']}°")
        print(f"   Worst performing angle: {summary['performance_statistics']['improvement_analysis']['worst_angle']}°")
        
        return summary
    else:
        print("❌ No successful results to summarize")
        return None

def main():
    """Main batch testing function"""
    print("🔬 NMF Angle Range Testing: 80-150 degrees")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Define angle range
    angles = [80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150]
    
    print(f"Testing {len(angles)} angles: {angles}")
    
    # Test all angles
    all_results = []
    total_start = time.time()
    
    for i, angle in enumerate(angles, 1):
        print(f"\n[{i}/{len(angles)}] Testing angle {angle}°...")
        result = test_single_angle(angle)
        all_results.append(result)
        
        # Progress update
        elapsed = time.time() - total_start
        if i < len(angles):
            estimated_total = elapsed * len(angles) / i
            remaining = estimated_total - elapsed
            print(f"   Progress: {i}/{len(angles)} ({i/len(angles)*100:.1f}%)")
            print(f"   Estimated remaining: {remaining/60:.1f} minutes")
    
    total_time = time.time() - total_start
    
    print(f"\n{'='*60}")
    print(f"All angle tests completed in {total_time/60:.1f} minutes")
    print('='*60)
    
    # Generate visualizations
    viz_results = generate_visualizations(all_results)
    
    # Create summary report
    summary = create_summary_report(all_results, viz_results)
    
    print(f"\n✅ Batch testing complete!")
    print(f"📁 Summary report: angle_test_summary.json")
    print(f"📁 Visualizations: angle_visualizations/")
    
    return all_results, viz_results, summary

if __name__ == '__main__':
    results, viz, summary = main()