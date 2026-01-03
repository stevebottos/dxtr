Analyze this Python module and extract semantic tags and a rich summary for retrieval purposes.

**Tags** should be high-level concepts that describe what this code does, NOT code identifiers:
- Domain tags: "computer vision", "NLP", "reinforcement learning", "data processing"
- Technique tags: "transfer learning", "fine-tuning", "batch processing", "embedding generation"
- Framework tags: "PyTorch", "Hugging Face Transformers", "scikit-learn"
- Architecture tags: "vision transformer", "encoder-decoder", "CNN", "GAN"
- Task tags: "image classification", "object detection", "text generation", "training loop"

Do NOT include: variable names, function names, class names, or low-level identifiers like "torch.nn" or "self.model"

**Summary** should richly describe:
- What the module does (purpose and functionality)
- Key techniques or patterns used
- How it fits into a larger system (if apparent)

Output JSON with:
- "keywords": array of 5-15 semantic tags (high-level concepts only)
- "summary": detailed 3-5 sentence description of the module's purpose and approach
