import ast

def extract_chunks_from_code(code: str, filepath: str):
    """
    Parses one Python file using AST and returns structured code chunks:
    1. Top-level module code (constants, global variables, imports, script blocks).
    2. Class definitions (with class name).
    3. Function & Method definitions (with ClassName.method_name parent scope context).
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    chunks = []
    top_level_nodes = []

    for stmt in tree.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Standalone Function
            docstring = ast.get_docstring(stmt) or "No docstring."
            source_code = ast.get_source_segment(code, stmt)
            if source_code:
                chunks.append({
                    "text": f"File: {filepath}\nName: {stmt.name}\nDocstring: {docstring}\n\n{source_code}",
                    "source": filepath,
                    "name": stmt.name
                })

        elif isinstance(stmt, ast.ClassDef):
            # Class definition
            class_name = stmt.name
            class_docstring = ast.get_docstring(stmt) or "No docstring."
            class_source = ast.get_source_segment(code, stmt)
            if class_source:
                chunks.append({
                    "text": f"File: {filepath}\nName: {class_name}\nDocstring: {class_docstring}\n\n{class_source}",
                    "source": filepath,
                    "name": class_name
                })

            # Extract class methods with ClassName.method_name scope context
            for item in stmt.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_name = f"{class_name}.{item.name}"
                    method_doc = ast.get_docstring(item) or "No docstring."
                    method_source = ast.get_source_segment(code, item)
                    if method_source:
                        chunks.append({
                            "text": f"File: {filepath}\nName: {method_name}\nDocstring: {method_doc}\n\n{method_source}",
                            "source": filepath,
                            "name": method_name
                        })
        else:
            # Collect top-level code (imports, constants, global assignments, script blocks)
            top_level_nodes.append(stmt)

    # If top-level code exists, create a <module_overview> chunk
    if top_level_nodes:
        top_level_segments = []
        for node in top_level_nodes:
            segment = ast.get_source_segment(code, node)
            if segment:
                top_level_segments.append(segment)

        if top_level_segments:
            top_level_text = "\n".join(top_level_segments).strip()
            # Only add if there is actual code (not just empty lines)
            if top_level_text:
                module_docstring = ast.get_docstring(tree) or "Top-level module code and constants."
                chunks.append({
                    "text": f"File: {filepath}\nName: <module_overview>\nDocstring: {module_docstring}\n\n{top_level_text}",
                    "source": filepath,
                    "name": "<module_overview>"
                })

    return chunks