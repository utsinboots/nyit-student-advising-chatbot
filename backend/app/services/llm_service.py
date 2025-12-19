"""
Complete Multi-Provider LLM Service
Supports: OpenAI, Anthropic (Claude), and Groq
"""

import time
from typing import Dict, Optional
import openai
from anthropic import Anthropic
from groq import Groq

from app.config import settings


class LLMService:
    """Multi-provider LLM service supporting OpenAI, Claude, and Groq"""
    
    def __init__(self):
        # OpenAI client
        if settings.OPENAI_API_KEY:
            openai.api_key = settings.OPENAI_API_KEY
            self.openai_available = True
        else:
            self.openai_available = False
        
        # Anthropic/Claude client
        if settings.ANTHROPIC_API_KEY:
            self.anthropic_client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)
            self.anthropic_available = True
        else:
            self.anthropic_client = None
            self.anthropic_available = False
        
        # Groq client
        if settings.GROQ_API_KEY:
            self.groq_client = Groq(api_key=settings.GROQ_API_KEY)
            self.groq_available = True
        else:
            self.groq_client = None
            self.groq_available = False
        
        # Usage tracking
        self.usage_stats = {
            'total_requests': 0,
            'total_tokens': 0,
            'total_cost': 0.0,
            'by_model': {}
        }
    
    def generate_response(
        self,
        query: str,
        context: str = "",
        model: str = None,
        max_tokens: int = 500,
        temperature: float = 0.7
    ) -> Dict:
        """
        Generate response using specified model.
        
        Auto-detects provider based on model name:
        - gpt-* → OpenAI
        - claude-* → Anthropic
        - llama-*, mixtral-* → Groq
        """
        
        # Use default model if not specified
        if not model:
            model = settings.CHAT_MODEL
        
        # Route to appropriate provider
        if model.startswith('claude'):
            return self._generate_claude(query, context, model, max_tokens, temperature)
        elif model.startswith('llama') or model.startswith('mixtral'):
            return self._generate_groq(query, context, model, max_tokens, temperature)
        else:
            return self._generate_openai(query, context, model, max_tokens, temperature)
    
    def _generate_openai(self, query, context, model, max_tokens, temperature):
        """Generate response using OpenAI"""
        
        if not self.openai_available:
            raise ValueError("OpenAI API key not configured")
        
        start_time = time.time()
        
        # Build messages
        system_prompt = """You are an academic advisor for NYIT's Computer Science M.S. program. 
Provide helpful, accurate information about courses, requirements, and academic policies.
Be concise but thorough."""
        
        if context:
            system_prompt += f"\n\nRelevant information:\n{context}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        
        try:
            response = openai.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            answer = response.choices[0].message.content
            
            tokens_used = {
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens
            }
            
            cost = self._calculate_cost_openai(model, tokens_used)
            self._update_stats(model, tokens_used['total_tokens'], cost)
            
            return {
                'answer': answer,
                'model_used': model,
                'tokens_used': tokens_used,
                'cost': cost,
                'latency_ms': latency_ms,
                'provider': 'openai'
            }
            
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")
    
    def _generate_claude(self, query, context, model, max_tokens, temperature):
        """Generate response using Claude/Anthropic"""
        
        if not self.anthropic_available:
            raise ValueError("Anthropic API key not configured")
        
        start_time = time.time()
        
        system_prompt = """You are an academic advisor for NYIT's Computer Science M.S. program. 
Provide helpful, accurate information about courses, requirements, and academic policies.
Be concise but thorough."""
        
        if context:
            system_prompt += f"\n\nRelevant information:\n{context}"
        
        try:
            response = self.anthropic_client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": query}]
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            answer = response.content[0].text
            
            tokens_used = {
                'prompt_tokens': response.usage.input_tokens,
                'completion_tokens': response.usage.output_tokens,
                'total_tokens': response.usage.input_tokens + response.usage.output_tokens
            }
            
            cost = self._calculate_cost_claude(model, tokens_used)
            self._update_stats(model, tokens_used['total_tokens'], cost)
            
            return {
                'answer': answer,
                'model_used': model,
                'tokens_used': tokens_used,
                'cost': cost,
                'latency_ms': latency_ms,
                'provider': 'anthropic'
            }
            
        except Exception as e:
            raise Exception(f"Anthropic API error: {str(e)}")
    
    def _generate_groq(self, query, context, model, max_tokens, temperature):
        """Generate response using Groq"""
        
        if not self.groq_available:
            raise ValueError("Groq API key not configured")
        
        start_time = time.time()
        
        system_prompt = """You are an academic advisor for NYIT's Computer Science M.S. program. 
Provide helpful, accurate information about courses, requirements, and academic policies.
Be concise but thorough."""
        
        if context:
            system_prompt += f"\n\nRelevant information:\n{context}"
        
        try:
            response = self.groq_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            answer = response.choices[0].message.content
            
            tokens_used = {
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens
            }
            
            cost = self._calculate_cost_groq(model, tokens_used)
            self._update_stats(model, tokens_used['total_tokens'], cost)
            
            return {
                'answer': answer,
                'model_used': model,
                'tokens_used': tokens_used,
                'cost': cost,
                'latency_ms': latency_ms,
                'provider': 'groq'
            }
            
        except Exception as e:
            raise Exception(f"Groq API error: {str(e)}")
    
    def _calculate_cost_openai(self, model, tokens_used):
        """Calculate cost for OpenAI models"""
        pricing = {
            'gpt-4o-mini': {'input': 0.15/1_000_000, 'output': 0.60/1_000_000},
            'gpt-4o': {'input': 2.50/1_000_000, 'output': 10.00/1_000_000},
            'gpt-4-turbo': {'input': 10.00/1_000_000, 'output': 30.00/1_000_000},
            'gpt-4': {'input': 30.00/1_000_000, 'output': 60.00/1_000_000},
            'gpt-3.5-turbo': {'input': 0.50/1_000_000, 'output': 1.50/1_000_000},
        }
        
        if model not in pricing:
            model = 'gpt-4o-mini'
        
        input_cost = tokens_used['prompt_tokens'] * pricing[model]['input']
        output_cost = tokens_used['completion_tokens'] * pricing[model]['output']
        
        return input_cost + output_cost
    
    def _calculate_cost_claude(self, model, tokens_used):
        """Calculate cost for Claude models"""
        pricing = {
            'claude-sonnet-4-20250514': {'input': 3.00/1_000_000, 'output': 15.00/1_000_000},
            'claude-3-5-sonnet-20241022': {'input': 3.00/1_000_000, 'output': 15.00/1_000_000},
            'claude-opus-4-20250514': {'input': 15.00/1_000_000, 'output': 75.00/1_000_000},
        }
        
        if model not in pricing:
            model = 'claude-sonnet-4-20250514'
        
        input_cost = tokens_used['prompt_tokens'] * pricing[model]['input']
        output_cost = tokens_used['completion_tokens'] * pricing[model]['output']
        
        return input_cost + output_cost
    
    def _calculate_cost_groq(self, model, tokens_used):
        """Calculate cost for Groq models"""
        pricing = {
            'llama-3.3-70b-versatile': {'input': 0.59/1_000_000, 'output': 0.79/1_000_000},
            'llama-3.1-70b-versatile': {'input': 0.59/1_000_000, 'output': 0.79/1_000_000},
            'llama3-70b-8192': {'input': 0.59/1_000_000, 'output': 0.79/1_000_000},
            'mixtral-8x7b-32768': {'input': 0.24/1_000_000, 'output': 0.24/1_000_000},
        }
        
        if model not in pricing:
            model = 'llama-3.3-70b-versatile'
        
        input_cost = tokens_used['prompt_tokens'] * pricing[model]['input']
        output_cost = tokens_used['completion_tokens'] * pricing[model]['output']
        
        return input_cost + output_cost
    
    def _update_stats(self, model, tokens, cost):
        """Update usage statistics"""
        self.usage_stats['total_requests'] += 1
        self.usage_stats['total_tokens'] += tokens
        self.usage_stats['total_cost'] += cost
        
        if model not in self.usage_stats['by_model']:
            self.usage_stats['by_model'][model] = {
                'requests': 0,
                'tokens': 0,
                'cost': 0.0
            }
        
        self.usage_stats['by_model'][model]['requests'] += 1
        self.usage_stats['by_model'][model]['tokens'] += tokens
        self.usage_stats['by_model'][model]['cost'] += cost
    
    def get_usage_stats(self):
        """Get current usage statistics"""
        return self.usage_stats