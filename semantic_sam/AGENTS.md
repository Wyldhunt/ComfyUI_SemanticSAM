# Semantic SAM Notes

## Instructions
- Preserve compatibility with existing SemanticSAM checkpoints and loading utilities.
- Keep tensor device transfers configurable; accept an optional device argument rather than calling `.cuda()` or `.cpu()` unconditionally inside helpers.
- When adding helper functions, document expected tensor shapes in the docstring.
