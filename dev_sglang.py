"""
SGLang stress test script.

This script validates that SGLang works correctly with the test model
and measures performance characteristics before migrating DXTR.

Prerequisites:
1. Start SGLang server: python start_sglang.py
2. Run this test: python dev_sglang.py
"""

import time
import concurrent.futures
from typing import List, Dict
import openai


def estimate_tokens(text: str) -> int:
    """Rough token estimation (actual tokens vary by tokenizer)."""
    # Rough approximation: ~1.3 tokens per word for English
    return int(len(text.split()) * 1.3)


class SGLangStressTester:
    """Stress tester for SGLang with gemma3 model."""

    def __init__(self, base_url: str = "http://localhost:30000/v1"):
        """Initialize tester with SGLang endpoint."""
        self.client = openai.Client(base_url=base_url, api_key="EMPTY")
        self.results: Dict[str, any] = {}

    def test_basic_completion(self) -> bool:
        """Test 1: Basic streaming completion."""
        print("\n" + "="*70)
        print("TEST 1: Basic Completion")
        print("="*70)

        start = time.time()
        try:
            print("Response: ", end="", flush=True)
            stream = self.client.chat.completions.create(
                model="default",
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": "Say 'Hello, I am working!' and nothing else."}
                ],
                max_tokens=50,
                temperature=0.0,
                stream=True
            )

            content = ""
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    delta = chunk.choices[0].delta.content
                    print(delta, end="", flush=True)
                    content += delta

            elapsed = time.time() - start
            print(f"\nTime: {elapsed:.2f}s")
            print("✓ PASSED")

            self.results['basic_completion'] = {
                'passed': True,
                'time': elapsed,
            }
            return True
        except Exception as e:
            print(f"✗ FAILED: {e}")
            self.results['basic_completion'] = {'passed': False, 'error': str(e)}
            return False

    def test_streaming(self) -> bool:
        """Test 2: Streaming response."""
        print("\n" + "="*70)
        print("TEST 2: Streaming Completion")
        print("="*70)

        start = time.time()
        try:
            stream = self.client.chat.completions.create(
                model="default",
                messages=[
                    {"role": "system", "content": "You are a concise AI assistant."},
                    {"role": "user", "content": "Count from 1 to 10, one number per line."}
                ],
                max_tokens=100,
                temperature=0.0,
                stream=True
            )

            print("Response: ", end="", flush=True)
            full_response = ""
            chunk_count = 0
            first_token_time = None

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    if first_token_time is None:
                        first_token_time = time.time() - start
                    content = chunk.choices[0].delta.content
                    print(content, end="", flush=True)
                    full_response += content
                    chunk_count += 1

            elapsed = time.time() - start
            print(f"\n\nTime to first token: {first_token_time:.2f}s")
            print(f"Total time: {elapsed:.2f}s")
            print(f"Chunks received: {chunk_count}")
            print("✓ PASSED")

            self.results['streaming'] = {
                'passed': True,
                'time': elapsed,
                'ttft': first_token_time,
                'chunks': chunk_count
            }
            return True
        except Exception as e:
            print(f"\n✗ FAILED: {e}")
            self.results['streaming'] = {'passed': False, 'error': str(e)}
            return False

    def test_long_context(self) -> bool:
        """Test 3: Long context handling (extended)."""
        print("\n" + "="*70)
        print("TEST 3: Long Context (Extended Multi-turn)")
        print("="*70)

        messages = [
            {"role": "system", "content": "You are a helpful research assistant who provides detailed explanations."}
        ]

        # Extended conversation to stress test context
        topics = [
            "Explain transformers in 2-3 sentences.",
            "How do they differ from RNNs? Give examples.",
            "What is the attention mechanism? Explain the query, key, value concept.",
            "Name three variants of transformers and their key differences.",
            "What are the computational complexity issues with transformers?",
            "How does positional encoding work in transformers?",
            "Explain the difference between encoder-only and decoder-only architectures.",
            "What innovations did GPT-3 introduce over GPT-2?",
        ]

        try:
            start = time.time()
            for i, topic in enumerate(topics, 1):
                messages.append({"role": "user", "content": topic})

                print(f"\nTurn {i}:")
                print(f"Q: {topic}")
                print(f"A: ", end="", flush=True)

                stream = self.client.chat.completions.create(
                    model="default",
                    messages=messages,
                    max_tokens=200,  # Allow longer responses
                    temperature=0.3,
                    stream=True
                )

                assistant_msg = ""
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        delta = chunk.choices[0].delta.content
                        print(delta, end="", flush=True)
                        assistant_msg += delta

                print()  # newline after response
                messages.append({"role": "assistant", "content": assistant_msg})

                # Show running context size
                context_so_far = "\n".join([m['content'] for m in messages])
                tokens_so_far = estimate_tokens(context_so_far)
                print(f"   [Context: ~{tokens_so_far:,} tokens]")

            elapsed = time.time() - start

            # Calculate context size
            full_context = "\n".join([m['content'] for m in messages])
            estimated_tokens = estimate_tokens(full_context)
            context_pct = (estimated_tokens / 32768) * 100

            print(f"\n\nTotal turns: {len(topics)}")
            print(f"Total time: {elapsed:.2f}s")
            print(f"Avg per turn: {elapsed/len(topics):.2f}s")
            print(f"Context length: ~{estimated_tokens:,} tokens ({context_pct:.1f}% of 32K)")
            print(f"Messages in context: {len(messages)}")
            print("✓ PASSED")

            self.results['long_context'] = {
                'passed': True,
                'time': elapsed,
                'turns': len(topics),
                'estimated_tokens': estimated_tokens,
                'context_pct': context_pct
            }
            return True
        except Exception as e:
            print(f"\n✗ FAILED: {e}")
            self.results['long_context'] = {'passed': False, 'error': str(e)}
            return False

    def _concurrent_request(self, request_id: int) -> Dict:
        """Helper for concurrent testing."""
        start = time.time()
        try:
            response = self.client.chat.completions.create(
                model="default",
                messages=[
                    {"role": "user", "content": f"Request {request_id}: What is 2+2? Answer with just the number."}
                ],
                max_tokens=10,
                temperature=0.0
            )
            elapsed = time.time() - start
            return {
                'id': request_id,
                'success': True,
                'time': elapsed,
                'response': response.choices[0].message.content.strip()
            }
        except Exception as e:
            elapsed = time.time() - start
            return {
                'id': request_id,
                'success': False,
                'time': elapsed,
                'error': str(e)
            }

    def test_long_context_concurrent(self, num_questions: int = 10) -> bool:
        """Test 4: Concurrent questions on shared long context."""
        print("\n" + "="*70)
        print(f"TEST 4: Long Context + Concurrent Questions (n={num_questions})")
        print("="*70)
        print("Testing RadixAttention prefix caching with shared context...")
        print()

        # Sample long document (simulating a research paper abstract/intro)
        long_document = """
TRANSFORMERS: ATTENTION IS ALL YOU NEED

Abstract:
The dominant sequence transduction models are based on complex recurrent or
convolutional neural networks that include an encoder and a decoder. The best
performing models also connect the encoder and decoder through an attention
mechanism. We propose a new simple network architecture, the Transformer,
based solely on attention mechanisms, dispensing with recurrence and convolutions
entirely. Experiments on two machine translation tasks show these models to be
superior in quality while being more parallelizable and requiring significantly
less time to train.

Introduction:
Recurrent neural networks (RNNs), long short-term memory (LSTM) networks, and
gated recurrent neural networks have been firmly established as state of the art
approaches in sequence modeling and transduction problems such as language modeling
and machine translation. Since then, numerous efforts have continued to push the
boundaries of recurrent language models and encoder-decoder architectures.

Attention mechanisms have become an integral part of compelling sequence modeling
and transduction models in various tasks, allowing modeling of dependencies without
regard to their distance in the input or output sequences. In all but a few cases,
however, such attention mechanisms are used in conjunction with a recurrent network.

In this work we propose the Transformer, a model architecture eschewing recurrence
and instead relying entirely on an attention mechanism to draw global dependencies
between input and output. The Transformer allows for significantly more parallelization
and can reach a new state of the art in translation quality after being trained for
as little as twelve hours on eight P100 GPUs.

Model Architecture:
Most competitive neural sequence transduction models have an encoder-decoder structure.
The encoder maps an input sequence to a sequence of continuous representations. The
decoder then generates an output sequence one element at a time. At each step the
model is auto-regressive, consuming the previously generated symbols as additional
input when generating the next.

The Transformer follows this overall architecture using stacked self-attention and
point-wise, fully connected layers for both the encoder and decoder.
""" * 3  # Repeat to make it longer (~2K tokens)

        # Questions to ask concurrently (all share same context prefix)
        questions = [
            "What is the main contribution of this paper?",
            "What did the authors replace RNNs with?",
            "How long did it take to train the model?",
            "What hardware was used for training?",
            "What are the two main components of the architecture?",
            "Why is the Transformer more parallelizable?",
            "What is the role of attention in this model?",
            "What previous approaches did the Transformer replace?",
            "What tasks were used to evaluate the model?",
            "What makes the Transformer auto-regressive?",
        ]

        context_tokens = estimate_tokens(long_document)
        print(f"Document loaded: ~{context_tokens:,} tokens")
        print(f"Asking {num_questions} concurrent questions with shared context...")
        print()

        def ask_question(q_id: int) -> Dict:
            """Ask a question about the document."""
            start = time.time()
            try:
                response = self.client.chat.completions.create(
                    model="default",
                    messages=[
                        {"role": "system", "content": f"You are analyzing this document:\n\n{long_document}"},
                        {"role": "user", "content": questions[q_id % len(questions)]}
                    ],
                    max_tokens=100,
                    temperature=0.0
                )
                elapsed = time.time() - start
                answer = response.choices[0].message.content.strip()
                return {
                    'id': q_id,
                    'success': True,
                    'time': elapsed,
                    'question': questions[q_id % len(questions)][:50],
                    'answer': answer[:80] + "..." if len(answer) > 80 else answer
                }
            except Exception as e:
                return {
                    'id': q_id,
                    'success': False,
                    'time': time.time() - start,
                    'error': str(e)
                }

        start = time.time()
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_questions) as executor:
                futures = [executor.submit(ask_question, i) for i in range(num_questions)]
                results = [f.result() for f in concurrent.futures.as_completed(futures)]

            elapsed = time.time() - start

            successes = sum(1 for r in results if r['success'])
            avg_latency = sum(r['time'] for r in results if r['success']) / max(successes, 1)
            min_latency = min((r['time'] for r in results if r['success']), default=0)
            max_latency = max((r['time'] for r in results if r['success']), default=0)

            print(f"Results:")
            print(f"  Questions answered: {successes}/{num_questions}")
            print(f"  Wall clock time: {elapsed:.2f}s")
            print(f"  Throughput: {num_questions/elapsed:.2f} q/s")
            print(f"  Latency - Avg: {avg_latency:.2f}s, Min: {min_latency:.2f}s, Max: {max_latency:.2f}s")
            print(f"  Context per request: ~{context_tokens:,} tokens (shared via RadixAttention)")
            print(f"\nSample Q&A:")

            for r in sorted(results, key=lambda x: x['id'])[:3]:
                if r['success']:
                    print(f"\n  Q: {r['question']}")
                    print(f"  A: {r['answer']}")

            passed = successes == num_questions
            print(f"\n{'✓ PASSED' if passed else '✗ FAILED'}")

            self.results['long_context_concurrent'] = {
                'passed': passed,
                'wall_time': elapsed,
                'questions': num_questions,
                'successes': successes,
                'context_tokens': context_tokens,
                'throughput': num_questions/elapsed,
                'avg_latency': avg_latency
            }
            return passed
        except Exception as e:
            print(f"\n✗ FAILED: {e}")
            self.results['long_context_concurrent'] = {'passed': False, 'error': str(e)}
            return False

    def test_concurrent_requests(self, num_users: int = 5) -> bool:
        """Test 5: Multi-tenant simulation with different user profiles."""
        print("\n" + "="*70)
        print(f"TEST 5: Multi-Tenant Simulation (n={num_users} users)")
        print("="*70)
        print("Simulating multiple users with different profiles asking questions...")
        print()

        # Different user profiles (simulating DXTR multi-tenancy)
        user_profiles = [
            {
                "name": "Alice",
                "profile": """You are helping Alice, a deep learning researcher.

Background: PhD in Computer Science, specializes in vision transformers and attention mechanisms.
Interests: Self-supervised learning, multimodal models, efficient architectures.
Recent work: Published papers on vision transformers for medical imaging.
Preferences: Prefers papers with novel architectural contributions and strong empirical results.""",
                "question": "What are the key innovations in this approach?"
            },
            {
                "name": "Bob",
                "profile": """You are helping Bob, a systems engineer in ML infrastructure.

Background: Works on distributed training systems and model serving at scale.
Interests: Training optimization, model compression, inference acceleration.
Recent work: Built multi-GPU training pipelines for LLMs.
Preferences: Interested in practical deployment considerations and performance benchmarks.""",
                "question": "How does this scale to large deployments?"
            },
            {
                "name": "Carol",
                "profile": """You are helping Carol, an NLP researcher focused on reasoning.

Background: PhD candidate studying chain-of-thought and reasoning in language models.
Interests: Emergent capabilities, in-context learning, reasoning tasks.
Recent work: Analyzing reasoning patterns in large language models.
Preferences: Papers that explore model capabilities and failure modes.""",
                "question": "What does this tell us about model reasoning?"
            },
            {
                "name": "David",
                "profile": """You are helping David, an ML researcher in robotics.

Background: Applies deep learning to robot control and perception.
Interests: Embodied AI, sim-to-real transfer, multi-task learning.
Recent work: Training policies for manipulation tasks using vision.
Preferences: Papers with real-world applications and physical grounding.""",
                "question": "How could this apply to robotics?"
            },
            {
                "name": "Eve",
                "profile": """You are helping Eve, a researcher in ML safety and alignment.

Background: Studies AI safety, interpretability, and robustness.
Interests: Model behavior understanding, failure analysis, safety mechanisms.
Recent work: Analyzing failure modes in production ML systems.
Preferences: Papers addressing robustness, interpretability, or safety concerns.""",
                "question": "What are potential risks or failure modes here?"
            },
        ]

        def user_query(user_id: int) -> Dict:
            """Simulate a user querying with their profile in context."""
            user = user_profiles[user_id % len(user_profiles)]
            start = time.time()

            try:
                response = self.client.chat.completions.create(
                    model="default",
                    messages=[
                        {"role": "system", "content": user["profile"]},
                        {"role": "user", "content": user["question"]}
                    ],
                    max_tokens=150,
                    temperature=0.3
                )
                elapsed = time.time() - start
                answer = response.choices[0].message.content.strip()

                return {
                    'user_id': user_id,
                    'user_name': user["name"],
                    'success': True,
                    'time': elapsed,
                    'profile_tokens': estimate_tokens(user["profile"]),
                    'question': user["question"],
                    'answer': answer[:100] + "..." if len(answer) > 100 else answer
                }
            except Exception as e:
                return {
                    'user_id': user_id,
                    'user_name': user["name"],
                    'success': False,
                    'time': time.time() - start,
                    'error': str(e)
                }

        start = time.time()
        try:
            # Simulate concurrent requests from different users
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_users) as executor:
                futures = [executor.submit(user_query, i) for i in range(num_users)]
                results = [f.result() for f in concurrent.futures.as_completed(futures)]

            elapsed = time.time() - start

            successes = sum(1 for r in results if r['success'])
            avg_latency = sum(r['time'] for r in results if r['success']) / max(successes, 1)
            min_latency = min((r['time'] for r in results if r['success']), default=0)
            max_latency = max((r['time'] for r in results if r['success']), default=0)

            # Get profile sizes
            avg_profile_tokens = sum(r.get('profile_tokens', 0) for r in results if r['success']) / max(successes, 1)

            print(f"Results:")
            print(f"  Users simulated: {num_users}")
            print(f"  Successful: {successes}")
            print(f"  Failed: {num_users - successes}")
            print(f"  Wall clock time: {elapsed:.2f}s")
            print(f"  Throughput: {num_users/elapsed:.2f} req/s")
            print(f"  Latency - Avg: {avg_latency:.2f}s, Min: {min_latency:.2f}s, Max: {max_latency:.2f}s")
            print(f"  Avg profile size: ~{int(avg_profile_tokens)} tokens")
            print(f"\nUser queries & responses:")

            for r in sorted(results, key=lambda x: x['user_id']):
                if r['success']:
                    print(f"\n  [{r['user_name']}] Profile: ~{r['profile_tokens']} tokens")
                    print(f"    Q: {r['question']}")
                    print(f"    A: {r['answer']}")
                    print(f"    Latency: {r['time']:.2f}s")

            passed = successes == num_users
            print(f"\n{'✓ PASSED' if passed else '✗ FAILED'}")

            self.results['concurrent'] = {
                'passed': passed,
                'wall_time': elapsed,
                'users': num_users,
                'successes': successes,
                'avg_profile_tokens': int(avg_profile_tokens),
                'avg_latency': avg_latency,
                'throughput': num_users/elapsed
            }
            return passed
        except Exception as e:
            print(f"\n✗ FAILED: {e}")
            self.results['concurrent'] = {'passed': False, 'error': str(e)}
            return False

    def test_tool_calling(self) -> bool:
        """Test 5: Tool/Function calling."""
        print("\n" + "="*70)
        print("TEST 5: Tool Calling")
        print("="*70)

        # Define a simple tool
        tools = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city name"
                        }
                    },
                    "required": ["location"]
                }
            }
        }]

        try:
            response = self.client.chat.completions.create(
                model="default",
                messages=[{"role": "user", "content": "What's the weather in San Francisco?"}],
                tools=tools,
                temperature=0.0
            )

            message = response.choices[0].message

            if hasattr(message, 'tool_calls') and message.tool_calls:
                tool_call = message.tool_calls[0]
                print(f"Tool called: {tool_call.function.name}")
                print(f"Arguments: {tool_call.function.arguments}")
                print("\n✓ PASSED - Tool calling works!")

                self.results['tool_calling'] = {
                    'passed': True,
                    'tool_name': tool_call.function.name,
                    'arguments': tool_call.function.arguments
                }
                return True
            else:
                print(f"Response: {message.content}")
                print("\n✗ FAILED - No tool call made (model may not support tools)")
                self.results['tool_calling'] = {
                    'passed': False,
                    'error': 'No tool call in response'
                }
                return False
        except Exception as e:
            print(f"\n✗ FAILED: {e}")
            self.results['tool_calling'] = {'passed': False, 'error': str(e)}
            return False

    def test_temperature_variation(self) -> bool:
        """Test 6: Temperature parameter effects."""
        print("\n" + "="*70)
        print("TEST 6: Temperature Variation")
        print("="*70)

        prompt = "Complete this sentence with one word: The sky is"
        temperatures = [0.0, 0.5, 1.0]

        try:
            results = []
            for temp in temperatures:
                print(f"Temperature {temp}: '", end="", flush=True)

                stream = self.client.chat.completions.create(
                    model="default",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=5,
                    temperature=temp,
                    stream=True
                )

                content = ""
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        delta = chunk.choices[0].delta.content
                        print(delta, end="", flush=True)
                        content += delta

                content = content.strip()
                print("'")
                results.append((temp, content))

            print("\n✓ PASSED (variations observed)" if len(set(r[1] for r in results)) > 1
                  else "\n⚠ PASSED (but no variation - might be expected for simple prompt)")

            self.results['temperature'] = {
                'passed': True,
                'results': results
            }
            return True
        except Exception as e:
            print(f"\n✗ FAILED: {e}")
            self.results['temperature'] = {'passed': False, 'error': str(e)}
            return False

    def run_all_tests(self) -> bool:
        """Run all stress tests."""
        print("\n" + "="*70)
        print("SGLang Stress Test Suite")
        print("="*70)
        print("This will validate SGLang functionality before DXTR migration.")
        print("="*70)

        tests = [
            self.test_basic_completion,
            self.test_streaming,
            self.test_long_context,
            self.test_long_context_concurrent,
            self.test_concurrent_requests,
            self.test_tool_calling,
            self.test_temperature_variation,
        ]

        results = [test() for test in tests]

        # Summary
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        passed = sum(results)
        total = len(results)
        print(f"Tests passed: {passed}/{total}")

        if passed == total:
            print("\n✓ ALL TESTS PASSED - SGLang is ready for DXTR migration!")
        else:
            print(f"\n✗ {total - passed} test(s) failed - review output above")

        print("="*70)
        return passed == total


def main():
    """Run the stress test suite."""
    tester = SGLangStressTester()

    try:
        success = tester.run_all_tests()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        print("\nMake sure SGLang server is running:")
        print("  python start_sglang.py")
        exit(1)


if __name__ == "__main__":
    main()
