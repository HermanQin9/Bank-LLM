"""Prompt optimization utilities."""

from typing import Dict, List, Optional, Tuple
from src.llm_engine.gemini_client import GeminiClient
from src.utils import logger


class PromptOptimizer:
    """Optimize prompts for better performance and cost efficiency."""
    
    def __init__(self, client: Optional[GeminiClient] = None):
        """
        Initialize prompt optimizer.
        
        Args:
            client: GeminiClient instance (creates new one if not provided)
        """
        self.client = client or GeminiClient()
        self.logger = logger
    
    def test_prompt_variations(
        self,
        prompt_variations: List[str],
        evaluation_criteria: str
    ) -> List[Dict[str, any]]:
        """
        Test multiple prompt variations and compare results.
        
        Args:
            prompt_variations: List of prompt variations to test
            evaluation_criteria: Criteria for evaluation
            
        Returns:
            List of results with scores
        """
        results = []
        
        for i, prompt in enumerate(prompt_variations):
            self.logger.info(f"Testing prompt variation {i+1}/{len(prompt_variations)}")
            
            response = self.client.generate_with_metadata(prompt)
            
            # Evaluate response
            eval_prompt = f"""Evaluate the following response based on these criteria: {evaluation_criteria}

Response:
{response['text']}

Provide a score from 1-10 and brief justification.

Score:"""
            
            evaluation = self.client.generate(eval_prompt)
            
            results.append({
                'prompt': prompt,
                'response': response['text'],
                'evaluation': evaluation,
                'tokens': response.get('response_tokens', 0)
            })
        
        return results
    
    def optimize_token_usage(self, prompt: str, max_reduction: float = 0.3) -> str:
        """
        Optimize prompt to reduce token usage while maintaining effectiveness.
        
        Args:
            prompt: Original prompt
            max_reduction: Maximum token reduction target (0-1)
            
        Returns:
            Optimized prompt
        """
        optimization_prompt = f"""Rewrite the following prompt to be more concise while maintaining clarity and effectiveness.
Reduce length by approximately {int(max_reduction * 100)}% without losing key information.

Original Prompt:
{prompt}

Optimized Prompt:"""
        
        optimized = self.client.generate(optimization_prompt, temperature=0.3)
        
        original_tokens = self.client.count_tokens(prompt)
        optimized_tokens = self.client.count_tokens(optimized)
        
        self.logger.info(f"Token reduction: {original_tokens} -> {optimized_tokens} ({(1 - optimized_tokens/original_tokens)*100:.1f}%)")
        
        return optimized
    
    def add_few_shot_examples(
        self,
        prompt: str,
        examples: List[Tuple[str, str]]
    ) -> str:
        """
        Add few-shot examples to prompt.
        
        Args:
            prompt: Base prompt
            examples: List of (input, output) example tuples
            
        Returns:
            Prompt with few-shot examples
        """
        examples_text = "\n\n".join([
            f"Example {i+1}:\nInput: {inp}\nOutput: {out}"
            for i, (inp, out) in enumerate(examples)
        ])
        
        enhanced_prompt = f"""Here are some examples:

{examples_text}

Now, please complete the following task:

{prompt}"""
        
        return enhanced_prompt
    
    def add_chain_of_thought(self, prompt: str) -> str:
        """
        Add chain-of-thought reasoning to prompt.
        
        Args:
            prompt: Base prompt
            
        Returns:
            Prompt with chain-of-thought instruction
        """
        cot_prompt = f"""{prompt}

Please think through this step-by-step:
1. First, analyze the key information
2. Then, reason through the solution
3. Finally, provide your answer

Let's work through this carefully:
"""
        
        return cot_prompt
    
    def create_structured_output_prompt(
        self,
        prompt: str,
        output_format: Dict[str, str]
    ) -> str:
        """
        Create prompt that requests structured output.
        
        Args:
            prompt: Base prompt
            output_format: Dictionary defining output structure
            
        Returns:
            Prompt with structured output format
        """
        format_description = "\n".join([
            f"{key}: {description}"
            for key, description in output_format.items()
        ])
        
        structured_prompt = f"""{prompt}

Please provide your response in the following structured format:

{format_description}

Response:
"""
        
        return structured_prompt
