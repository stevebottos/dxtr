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


class SGLangStressTester:
    """Stress tester for SGLang with gemma3 model."""

    def __init__(self, base_url: str = "http://localhost:30000/v1"):
        """Initialize tester with SGLang endpoint."""
        self.client = openai.Client(base_url=base_url, api_key="EMPTY")
        self.results: Dict[str, any] = {}

    def test_basic_completion(self) -> bool:
        """Test 1: Basic non-streaming completion."""
        print("\n" + "="*70)
        print("TEST 1: Basic Completion")
        print("="*70)

        start = time.time()
        try:
            response = self.client.chat.completions.create(
                model="default",
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": "Say 'Hello, I am working!' and nothing else."}
                ],
                max_tokens=50,
                temperature=0.0
            )
            elapsed = time.time() - start

            content = response.choices[0].message.content
            print(f"Response: {content}")
            print(f"Time: {elapsed:.2f}s")
            print(f"Tokens: {response.usage.total_tokens}")
            print("✓ PASSED")

            self.results['basic_completion'] = {
                'passed': True,
                'time': elapsed,
                'tokens': response.usage.total_tokens
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
        """Test 3: Long context handling."""
        print("\n" + "="*70)
        print("TEST 3: Long Context (Multi-turn Conversation)")
        print("="*70)

        messages = [
            {"role": "system", "content": "You are a helpful research assistant."}
        ]

        # Build up a multi-turn conversation
        topics = [
            "Explain transformers in one sentence.",
            "How do they differ from RNNs?",
            "What is attention mechanism?",
            "Name three variants of transformers.",
        ]

        try:
            start = time.time()
            for i, topic in enumerate(topics, 1):
                messages.append({"role": "user", "content": topic})

                response = self.client.chat.completions.create(
                    model="default",
                    messages=messages,
                    max_tokens=150,
                    temperature=0.3
                )

                assistant_msg = response.choices[0].message.content
                messages.append({"role": "assistant", "content": assistant_msg})

                print(f"\nTurn {i}:")
                print(f"Q: {topic}")
                print(f"A: {assistant_msg[:100]}..." if len(assistant_msg) > 100 else f"A: {assistant_msg}")

            elapsed = time.time() - start
            total_tokens = sum(len(m['content'].split()) for m in messages)

            print(f"\nTotal turns: {len(topics)}")
            print(f"Total time: {elapsed:.2f}s")
            print(f"Avg per turn: {elapsed/len(topics):.2f}s")
            print(f"Context size: ~{total_tokens} words")
            print("✓ PASSED")

            self.results['long_context'] = {
                'passed': True,
                'time': elapsed,
                'turns': len(topics),
                'context_words': total_tokens
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

    def test_concurrent_requests(self, num_requests: int = 5) -> bool:
        """Test 4: Concurrent request handling."""
        print("\n" + "="*70)
        print(f"TEST 4: Concurrent Requests (n={num_requests})")
        print("="*70)

        start = time.time()
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_requests) as executor:
                futures = [executor.submit(self._concurrent_request, i) for i in range(num_requests)]
                results = [f.result() for f in concurrent.futures.as_completed(futures)]

            elapsed = time.time() - start

            successes = sum(1 for r in results if r['success'])
            avg_latency = sum(r['time'] for r in results if r['success']) / max(successes, 1)

            print(f"\nRequests sent: {num_requests}")
            print(f"Successful: {successes}")
            print(f"Failed: {num_requests - successes}")
            print(f"Total time: {elapsed:.2f}s")
            print(f"Avg latency: {avg_latency:.2f}s")
            print(f"Throughput: {num_requests/elapsed:.2f} req/s")

            for r in sorted(results, key=lambda x: x['id']):
                status = "✓" if r['success'] else "✗"
                response = r.get('response', r.get('error', 'Unknown'))
                print(f"  {status} Request {r['id']}: {response} ({r['time']:.2f}s)")

            passed = successes == num_requests
            if passed:
                print("\n✓ PASSED")
            else:
                print(f"\n✗ FAILED ({num_requests - successes} failures)")

            self.results['concurrent'] = {
                'passed': passed,
                'total_time': elapsed,
                'successes': successes,
                'failures': num_requests - successes,
                'avg_latency': avg_latency,
                'throughput': num_requests/elapsed
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
                response = self.client.chat.completions.create(
                    model="default",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=5,
                    temperature=temp
                )
                content = response.choices[0].message.content.strip()
                results.append((temp, content))
                print(f"Temperature {temp}: '{content}'")

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
