You are evaluating a generated code analysis against the actual source code.

The analysis contains semantic TAGS (not literal code identifiers) and a summary.

**Tags Score (1-5):**
- 5: All tags accurately describe the code's domain, techniques, and purpose
- 4: Most tags are relevant and accurate, minor issues
- 3: Some tags accurate, some irrelevant or slightly off
- 2: Many tags don't apply to this code or are hallucinated
- 1: Tags are mostly wrong or describe something this code doesn't do

Tags should be high-level concepts like "computer vision", "PyTorch", "training loop", "encoder-decoder".
Do NOT penalize for missing literal code matches - tags are semantic, not syntactic.

**Summary Score (1-5):**
- 5: Summary accurately and thoroughly describes what the code does
- 4: Summary mostly accurate, minor imprecisions or omissions
- 3: Summary partially correct but missing key aspects or has minor errors
- 2: Summary has significant inaccuracies about what the code does
- 1: Summary is mostly wrong or describes something the code doesn't do

For issues, be specific (e.g., "tag 'NLP' incorrect - this is computer vision code", "summary claims real-time processing but code is batch-only").

Output JSON with:
- "keywords_score": integer 1-5
- "keywords_issues": array of specific issues (empty if score is 5)
- "summary_score": integer 1-5
- "summary_issues": array of specific issues (empty if score is 5)
