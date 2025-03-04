##################### JUDGE PROMPTS #####################
#### SCHEMA GENERATION PROMPT
SCHEMA_GENERATION_JUDGE_PROMPT = """
You are a judge evaluating the similarity evaluating the similarity between a model-generated schema 
and a ground truth schema for a collection of unstructured documents. Your goal is to assess the semantic, 
structural, and categorical alignment between the two schemas.

Use the following criteria to arrive at a score:
1. Exact Matches: Identical fields in both schemas.
2. Partial Matches: Fields that are similar in meaning but differ in wording, granularity, or format.
3. Missing Fields: Fields present in the ground truth but absent in the model-generated schema.
4. Extra Fields: Fields in the model-generated schema but not in the ground truth.
5. Hierarchical/Structural Similarity: How well the nesting, grouping, or relationships match.
6. Semantic Consistency: Whether field names reflect the intended meaning correctly.

Provide a score between 0 and 1 in the following JSON format:
{
  "score": score between 0 and 1
}
Return only the score without any extra information. 
"""
