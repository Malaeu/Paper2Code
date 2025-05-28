"""Integrated LLM Evaluator combining existing utility framework with RAPI analysis"""

from typing import Dict, List, Any, Optional
import yaml
from utils import calculate_utility, filter_by_constraints
from rapi_calculator import calculate_RAPI_v2, get_model_data

class IntegratedLLMEvaluator:
    """Combines utility-based and RAPI-based evaluation methods"""
    
    def __init__(self, models_file: str = "models.yaml", config_file: str = "config.yaml"):
        self.models = self._load_models(models_file)
        self.config = self._load_config(config_file)
        self.rapi_models = get_model_data()
    
    def _load_models(self, filename: str) -> List[Dict[str, Any]]:
        """Load model data from YAML file"""
        with open(filename, 'r') as f:
            data = yaml.safe_load(f)
            return data['models']
    
    def _load_config(self, filename: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        with open(filename, 'r') as f:
            return yaml.safe_load(f)
    
    def evaluate_all_models(self, use_case: str = "default") -> List[Dict[str, Any]]:
        """Evaluate all models using both utility and RAPI metrics"""
        results = []
        
        # Get weights for the use case
        if use_case in self.config.get('profiles', {}):
            weights = self.config['profiles'][use_case]
        else:
            weights = self.config.get('weights', {
                'alpha': 1.0,
                'beta': 0.8,
                'gamma': 1.0,
                'delta': 1.2
            })
        
        for model in self.models:
            # Calculate utility score
            utility_score = calculate_utility(model, weights)
            
            # Calculate RAPI if model data available
            rapi_score = None
            model_key = self._get_model_key(model['name'])
            
            if model_key in self.rapi_models:
                rapi_data = self._convert_to_rapi_format(model, self.rapi_models[model_key])
                
                # Handle models with reasoning modes
                if 'reasoning_modes' in model:
                    rapi_scores = {}
                    for mode in ['low', 'medium', 'high']:
                        rapi_scores[mode] = calculate_RAPI_v2(rapi_data, mode)
                    rapi_score = rapi_scores
                else:
                    rapi_score = calculate_RAPI_v2(rapi_data)
            
            results.append({
                'name': model['name'],
                'utility_score': round(utility_score, 3),
                'rapi_score': rapi_score,
                'quality': model['quality'],
                'speed': model['speed'],
                'stability': model['stability'],
                'cost_penalty': model['cost_penalty'],
                'context_window': model.get('context_window', 128000)
            })
        
        # Sort by utility score
        results.sort(key=lambda x: x['utility_score'], reverse=True)
        return results
    
    def _get_model_key(self, model_name: str) -> str:
        """Convert model name to key for RAPI lookup"""
        mappings = {
            'Claude 3.5 Sonnet': 'claude_3.5_sonnet',
            'Gemini 2.5 Flash': 'gemini_2.5_flash',
            'GPT-4.1': 'gpt-4.1',
            'o4-mini': 'o4-mini'
        }
        return mappings.get(model_name, model_name.lower().replace(' ', '_'))
    
    def _convert_to_rapi_format(self, utility_model: Dict, rapi_model: Dict) -> Dict:
        """Convert utility model format to RAPI format"""
        # Merge data from both sources
        return {
            'quality': utility_model['quality'],
            'latency': rapi_model.get('latency', 2.0),
            'speed_tps': rapi_model.get('speed_tps', 100),
            'context_window': utility_model.get('context_window', rapi_model.get('context_window', 128000)),
            'input_cost_per_m': rapi_model.get('input_cost_per_m', utility_model['cost_penalty'] * 20),
            'output_cost_per_m': rapi_model.get('output_cost_per_m', utility_model['cost_penalty'] * 80),
            'latency_modes': rapi_model.get('latency_modes', {})
        }
    
    def get_optimal_model(self, use_case: str, constraints: Optional[Dict] = None) -> Dict[str, Any]:
        """Get the optimal model for a specific use case with constraints"""
        all_models = self.evaluate_all_models(use_case)
        
        # Apply constraints if provided
        if constraints:
            filtered = []
            for model in all_models:
                if constraints.get('min_quality') and model['quality'] < constraints['min_quality']:
                    continue
                if constraints.get('max_cost_penalty') and model['cost_penalty'] > constraints['max_cost_penalty']:
                    continue
                if constraints.get('min_stability') and model['stability'] < constraints['min_stability']:
                    continue
                if constraints.get('min_context_window') and model['context_window'] < constraints['min_context_window']:
                    continue
                filtered.append(model)
            all_models = filtered
        
        return all_models[0] if all_models else None
    
    def generate_comparison_report(self) -> str:
        """Generate comprehensive comparison report"""
        report = ["# Integrated LLM Evaluation Report\n"]
        
        # Evaluate for different use cases
        use_cases = ['quality_first', 'speed_optimized', 'budget_conscious', 'default']
        
        for use_case in use_cases:
            report.append(f"\n## {use_case.replace('_', ' ').title()} Rankings\n")
            results = self.evaluate_all_models(use_case)[:10]  # Top 10
            
            report.append("| Model | Utility | RAPI | Quality | Speed | Cost Penalty | Context |")
            report.append("|-------|---------|------|---------|-------|--------------|---------|")
            
            for r in results:
                rapi_str = str(r['rapi_score']) if isinstance(r['rapi_score'], (int, float)) else 'Multi-mode'
                context_str = f"{r['context_window']//1000}K" if r['context_window'] < 1000000 else f"{r['context_window']//1000000}M"
                
                report.append(f"| {r['name']} | {r['utility_score']:.3f} | {rapi_str} | "
                            f"{r['quality']:.2f} | {r['speed']:.2f} | {r['cost_penalty']:.3f} | {context_str} |")
        
        # Add insights section
        report.append("\n## Key Insights\n")
        report.append("1. **Utility vs RAPI**: Utility scoring considers user-defined weights, while RAPI provides absolute performance/cost ratio")
        report.append("2. **Context Window Impact**: Models with 1M+ context (GPT-4.1, Gemini 2.5) get significant RAPI boost")
        report.append("3. **Reasoning Modes**: o4-mini offers flexible quality/latency trade-offs not captured in simple utility scores")
        report.append("4. **Cost Leaders**: Gemini 2.5 Flash dominates on cost-efficiency across all metrics")
        
        return '\n'.join(report)


def main():
    """Run integrated evaluation"""
    evaluator = IntegratedLLMEvaluator()
    
    # Generate and save report
    report = evaluator.generate_comparison_report()
    with open('integrated_evaluation_report.md', 'w') as f:
        f.write(report)
    
    print("Integrated evaluation complete! Report saved to integrated_evaluation_report.md")
    
    # Show optimal models for different scenarios
    print("\n🎯 Optimal Models by Scenario:\n")
    
    scenarios = [
        ("High-quality code generation", "quality_first", {"min_quality": 0.85}),
        ("Real-time chat", "speed_optimized", {"max_cost_penalty": 0.1}),
        ("Long document processing", "default", {"min_context_window": 500000}),
        ("Budget-conscious reasoning", "budget_conscious", {"min_quality": 0.8})
    ]
    
    for scenario, use_case, constraints in scenarios:
        optimal = evaluator.get_optimal_model(use_case, constraints)
        if optimal:
            print(f"{scenario}: {optimal['name']} (Utility: {optimal['utility_score']:.3f})")


if __name__ == "__main__":
    main()