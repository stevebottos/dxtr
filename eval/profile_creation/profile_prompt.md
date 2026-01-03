You are evaluating a synthesized developer profile against the original profile and GitHub analysis data.

Score each aspect from 1-5:

**Accuracy (1-5):**
- 5: Profile accurately represents skills, projects, and experience from the GitHub analysis
- 4: Mostly accurate, minor misrepresentations
- 3: Some accurate content, but notable errors or exaggerations
- 2: Significant inaccuracies or fabricated skills/experience
- 1: Mostly fabricated or contradicts the source data

**Completeness (1-5):**
- 5: Covers all major skills, technologies, and projects from the analysis
- 4: Covers most important aspects, minor omissions
- 3: Covers some key areas but misses significant content
- 2: Missing most of the important skills or projects
- 1: Fails to represent the person's actual work

**Coherence (1-5):**
- 5: Well-structured, professional, reads naturally
- 4: Good structure, minor awkwardness
- 3: Adequate structure but some disjointed sections
- 2: Poorly organized or difficult to follow
- 1: Incoherent or unprofessional

For issues, be specific (e.g., "claims expertise in Rust but no Rust code in repos", "missing the computer vision project entirely").

Output JSON with:
- "accuracy_score": integer 1-5
- "accuracy_issues": array of specific issues (empty if score is 5)
- "completeness_score": integer 1-5
- "completeness_issues": array of specific issues (empty if score is 5)
- "coherence_score": integer 1-5
- "coherence_issues": array of specific issues (empty if score is 5)
