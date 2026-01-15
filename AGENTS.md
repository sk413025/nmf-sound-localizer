# Project Memory - Interspeech 2026 Manuscript

## 🚨 CRITICAL: Every Commit Must Be Verifiable

**MANDATORY REQUIREMENT**: Every paper commit MUST:
1. ✅ **Have been reviewed for accuracy** before committing
2. ✅ **Support claims with evidence** (data/citations)
3. ✅ **Document rationale** for changes
4. ✅ **Be self-contained** - single logical change per commit

**NO unverified claims. NO placeholder content. NO incomplete sections.**

## Commit Units — Manuscript Changes Only

- Granularity: Each commit = one logical change (section draft, figure update, reference batch)
- Atomicity: Commit related changes together (e.g., figure + text referencing it)
- Content requirements (must be explicit in the commit message body):
  - Background: What state/problem existed before this change?
  - Changes: What was added/modified/removed?
  - Rationale: Why this change improves the paper?
  - Evidence: How are claims supported?
  - Quality: Assessment of clarity, completeness, accuracy

### Commit Prefixes

| Prefix | Usage | Example |
|--------|-------|---------|
| `Paper:` | Major section drafts, rewrites, new content | `Paper: Introduction - Add problem formulation` |
| `Edit:` | Revisions, corrections, improvements | `Edit: Methods - Clarify frequency embedding` |
| `Figure:` | New or updated figures | `Figure: Add architecture diagram` |
| `Table:` | New or updated tables | `Table: Update results comparison` |
| `Ref:` | Reference additions or updates | `Ref: Add Decision Transformer citations` |
| `Response:` | Review response drafts | `Response: Address R1 concern on evaluation` |
| `Format:` | Layout, styling, formatting only | `Format: Fix equation numbering` |

Notes:
- If a change spans multiple categories, use the primary one
- For mixed changes, land separate commits per category when possible

## Commit Message Format (Required)

```
Paper: [Section] - Brief description

Background:
- What was the state before this change?
- What problem/gap existed?

Changes:
- What was added/modified/removed?
- Which files were affected?

Rationale:
- Why is this change needed?
- How does it improve the paper?

Evidence verification:
- [ ] All claims supported by data or citations
- [ ] No unsupported assertions

Cross-section consistency:
- Related sections checked: [list sections]
- Terminology verified consistent

Quality assessment:
- Clarity: [rating or comment]
- Completeness: [what's still missing]
- Accuracy: [verification status]
```

## 🎯 Writing Simplification Philosophy

**CRITICAL: Every revision should prioritize clarity and reduce complexity**

### Core Principles
1. **Clarity First**: Prefer simple sentences over complex constructions
2. **Remove Redundancy**: Cut words that don't add meaning
3. **One Idea Per Sentence**: Don't overload sentences
4. **Active Voice**: Prefer "We propose" over "It is proposed"
5. **Concrete Over Abstract**: Use specific examples and numbers

### Before Writing, Ask:
- Can this be said in fewer words?
- Is every sentence necessary?
- Would a reviewer understand this on first read?
- Are there redundant explanations?
- Does this add to the paper's core message?

### Prefer:
- Short paragraphs over long blocks
- Bullet points for lists of items
- Tables for comparisons
- Figures for complex relationships
- Direct statements over hedging ("achieves" not "appears to achieve")
- Specific numbers over vague descriptions ("97.11%" not "near-optimal")

### Avoid:
- Unnecessary adjectives ("novel", "unique", "significant")
- Hedge words without justification ("may", "might", "could")
- Redundant phrases ("in order to" → "to")
- Passive voice when active is clearer
- Long parenthetical insertions

## Quality Checklist for Paper Commits

### Content Quality
- [ ] Claims are supported by experimental results or citations
- [ ] Mathematical notation is consistent (e.g., always $\Delta\phi$, not Δφ)
- [ ] Figures/tables are referenced in text
- [ ] Figure captions are complete and self-explanatory
- [ ] Table formatting follows Interspeech guidelines
- [ ] No placeholder content (TODO, TBD, XXX)

### Writing Quality
- [ ] No grammar or spelling errors
- [ ] Terminology is consistent throughout
- [ ] Acronyms defined on first use (LDV, OMP, DT, etc.)
- [ ] Passive/active voice used consistently
- [ ] Sentences are concise and clear
- [ ] Technical terms explained for general audience

### Technical Quality
- [ ] Equations are numbered if referenced
- [ ] Variables defined before use
- [ ] Units specified where applicable (Hz, dB, etc.)
- [ ] Results match source data/experiments
- [ ] Hyperparameters fully specified
- [ ] Reproducibility information complete

### Format Quality
- [ ] Page limit respected (4 pages + references for Interspeech)
- [ ] References complete (no missing fields)
- [ ] Anonymous for submission (no author names/affiliations)
- [ ] Figures high resolution (300+ DPI)
- [ ] Font sizes readable in figures
- [ ] Consistent citation style

## 🔴 Mandatory Reflection Requirements

**Every significant paper commit MUST include ALL sections below.**

### 1. Content Analysis
- What key message does this section convey?
- How does it connect to the paper's central thesis?
- What evidence supports the claims made?
- What is the "so what" - why should readers care?

### 2. Cross-Section Analysis
- How does this change affect other sections?
- What forward/backward references need updating?
- Is terminology consistent with other sections?
- Does the narrative flow logically from previous sections?

### 3. Reader Perspective
- Is the logic flow clear to a first-time reader?
- Are assumptions stated explicitly?
- Would a reviewer find this convincing?
- What questions might a reviewer ask?
- Are potential weaknesses acknowledged?

### 4. Improvement Principles
- What writing patterns work well here?
- What to avoid in future revisions?
- What style choices should be replicated?
- How could this section be further improved?

## Language Policy

**All project content must be written in English**, including:
- Manuscript text and comments
- Git commit messages
- Project memory (this file)
- Figure labels and captions
- Code comments in supporting scripts

## Essential Commands

```bash
# Build PDF from markdown
cd paper && make

# Or with pandoc directly
pandoc draft.md -o draft.pdf \
    --metadata-file=metadata.yaml \
    --citeproc \
    --bibliography=references.bib

# Word count check (approximate)
pandoc draft.md -t plain | wc -w

# Character count for abstract (Interspeech limit: ~1000 chars)
sed -n '/^abstract:/,/^---/p' paper/draft.md | wc -c

# Check for incomplete markers
grep -n "TODO\|TBD\|FIXME\|XXX" paper/draft.md paper/references.bib

# Validate BibTeX
biber --tool --validate-datamodel references.bib

# Check for undefined references
pandoc draft.md --citeproc --bibliography=references.bib 2>&1 | grep -i "undefined"
```

## Paper Structure

```
paper/
├── draft.md           # Main manuscript (Markdown + LaTeX math)
├── metadata.yaml      # Pandoc settings (title, format, packages)
├── references.bib     # BibTeX references
├── Makefile           # Build automation
└── figures/
    ├── architecture.png       # Model architecture diagram
    ├── results_comparison.png # Main results visualization
    ├── freq_embedding_sim.png # Frequency embedding analysis
    └── ...
```

## Figure Management

### Naming Convention
- Descriptive names: `freq_embedding_sim.png` not `fig1.png`
- Lowercase with underscores
- Include version suffix if iterating: `results_v2.png`

### Quality Requirements
- Resolution: 300+ DPI for print
- Format: PNG for diagrams, PDF for vector graphics
- Size: Fit within column width (typically 3.5 inches)
- Fonts: Match paper font size (readable at final size)
- Colors: Consider colorblind accessibility

### Commit with Figures
When adding/updating figures, commit together with:
- The figure file itself
- Updated text referencing the figure
- Updated caption if changed

## Reference Management

### BibTeX Best Practices
- Use consistent key format: `author2024keyword`
- Include all required fields (author, title, year, venue)
- Use proper capitalization in titles: `{Decision Transformer}`
- Verify DOI links work

### Citation Style
- Use `[@key]` for parenthetical citations
- Use `@key` for narrative citations: "As shown by @author2024..."
- Group related citations: `[@key1; @key2; @key3]`

## Examples: Good vs Bad Commits

### ❌ Bad Commit
```
Update paper
```

### ❌ Bad Commit (no rationale)
```
Paper: Results - Add numbers

Added accuracy numbers to results table.
```

### ✅ Good Commit
```
Paper: Results - Add frequency-aware DT performance metrics

Background:
- Results section had placeholder values from initial experiments
- Missing comparison with single-bin baseline

Changes:
- Updated Table 2 with final experiment results
- Added single-bin (50-60) baseline row
- Updated prose to reflect 97.11% energy reduction

Rationale:
- Provides complete picture of model performance
- Single-bin comparison shows generalization benefit
- Numbers verified against experiment logs (results/freq_aware_full/metrics.json)

Evidence verification:
- [x] All numbers from verified experiment runs
- [x] Baseline comparisons use same test set

Cross-section consistency:
- Related sections checked: Abstract, Introduction, Conclusion
- Updated abstract to match final numbers

Quality assessment:
- Clarity: Good - table format makes comparison clear
- Completeness: Results section now complete
- Accuracy: Verified against source logs
```

## Interspeech 2026 Specifics

### Format Requirements
- Page limit: 4 pages (content) + unlimited references
- Paper size: A4
- Columns: Two-column layout
- Font: 10pt minimum
- Anonymous submission: No author names in initial submission

### Key Sections
1. **Introduction** (~0.75 page): Problem, motivation, contribution
2. **Method** (~1.25 pages): Architecture, training procedure
3. **Experiments** (~1.5 pages): Dataset, baselines, results, ablation
4. **Conclusion** (~0.5 page): Summary, limitations, future work

### Submission Checklist
- [ ] Page limit met
- [ ] Anonymous (no author info)
- [ ] All figures referenced
- [ ] All tables referenced
- [ ] References complete
- [ ] Abstract within character limit
- [ ] PDF generated and reviewed
- [ ] Supplementary materials prepared (if any)

## Related Experiment Data

This paper is based on experiments from the `feature/freq-aware-policy` branch. Key experiment commits:

- Frequency-aware architecture: See `results/freq_aware_smoke/`
- Full-scale evaluation: See `results/freq_aware_full/`
- Ablation studies: See `results/ablation_*/`

When citing experimental results, always verify numbers against the source logs in the `results/` directory.

## Last Updated
2025-01-15 (Converted from experiment tracking to paper writing guidelines)
