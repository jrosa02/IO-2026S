#!/usr/bin/env python3
"""
Generate UML diagrams from Python source files in src directory.
Uses ast module to parse Python files and graphviz to visualize.
"""

import ast
from pathlib import Path
from typing import Dict, List

import graphviz


class PythonAnalyzer(ast.NodeVisitor):
    """Analyze Python files to extract class and method information."""

    def __init__(self):
        self.classes: Dict[str, Dict] = {}
        self.current_class = None

    def visit_ClassDef(self, node):
        """Extract class definition."""
        bases = [self._get_name(base) for base in node.bases]
        self.classes[node.name] = {
            'bases': bases,
            'methods': [],
            'attributes': [],
        }
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = None

    def visit_FunctionDef(self, node):
        """Extract method definitions."""
        if self.current_class:
            # Get return type and parameters
            args = [arg.arg for arg in node.args.args if arg.arg != 'self']
            self.classes[self.current_class]['methods'].append({
                'name': node.name,
                'args': args,
            })
        self.generic_visit(node)

    def visit_Assign(self, node):
        """Extract class attributes (simple case)."""
        if self.current_class and isinstance(node.targets[0], ast.Name):
            attr_name = node.targets[0].id
            if not attr_name.startswith('_'):  # Skip private
                self.classes[self.current_class]['attributes'].append(attr_name)
        self.generic_visit(node)

    @staticmethod
    def _get_name(node):
        """Get the name from various node types."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return node.attr
        return str(node)


def collect_python_files(src_dir: str) -> List[Path]:
    """Find all Python files in src directory."""
    src_path = Path(src_dir)
    return sorted(src_path.glob('**/*.py'))


def analyze_python_files(files: List[Path]) -> Dict[str, Dict]:
    """Analyze all Python files and extract class information."""
    all_classes = {}

    for file_path in files:
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                tree = ast.parse(f.read())
                analyzer = PythonAnalyzer()
                analyzer.visit(tree)
                all_classes.update(analyzer.classes)
            except SyntaxError as e:
                print(f"Warning: Could not parse {file_path}: {e}")

    return all_classes


def extract_relationships(files: List[Path], classes: Dict[str, Dict]) -> Dict[str, set]:
    """Extract class relationships from type hints and instantiation patterns."""
    relationships = {class_name: set() for class_name in classes}

    for file_path in files:
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                content = f.read()
                tree = ast.parse(content)
            except SyntaxError:
                continue

            # Walk through AST nodes
            for node in ast.walk(tree):
                current_class = None

                # Find class definitions and their contents
                if isinstance(node, ast.ClassDef):
                    current_class = node.name
                    if current_class not in classes:
                        continue

                    # Check for method parameters with class types
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            # Check type hints in __init__ and other methods
                            for arg in item.args.args:
                                if arg.annotation:
                                    if isinstance(arg.annotation, ast.Name) and arg.annotation.id in classes:
                                        if arg.annotation.id != current_class:
                                            relationships[current_class].add(arg.annotation.id)
                                    elif isinstance(arg.annotation, ast.Subscript):
                                        # Handle generic types like List[SchInstance]
                                        if isinstance(arg.annotation.value, ast.Name):
                                            base_type = arg.annotation.value.id
                                            if base_type in classes:
                                                relationships[current_class].add(base_type)
                                        if isinstance(arg.annotation.slice, ast.Name):
                                            inner_type = arg.annotation.slice.id
                                            if inner_type in classes:
                                                relationships[current_class].add(inner_type)

                            # Check assignments in method bodies (e.g., self.net = ActorCritic(...))
                            for body_node in ast.walk(item):
                                if isinstance(body_node, ast.Call):
                                    if isinstance(body_node.func, ast.Name) and body_node.func.id in classes:
                                        if body_node.func.id != current_class:
                                            relationships[current_class].add(body_node.func.id)

    return relationships


def generate_graphviz_diagram(classes: Dict[str, Dict], relationships: Dict[str, set] | None = None) -> graphviz.Digraph:
    """Generate Graphviz UML diagram with proper styling."""
    dot = graphviz.Digraph(comment='Python UML Diagram', format='svg', engine='dot')
    dot.attr(rankdir='TB', splines='spline', overlap='false')
    dot.attr('node', shape='plaintext', fontname='Courier', fontsize='9')
    dot.attr('graph', bgcolor='#f9f9f9')

    # Create nodes for each class
    for class_name, info in sorted(classes.items()):
        # Build class box in record shape
        attrs = []
        if info['attributes']:
            attrs = info['attributes'][:5]  # Limit to 5 to avoid huge boxes

        methods = [m['name'] for m in info['methods'][:8]]  # Limit to 8 methods

        # Create label with proper record structure
        attr_str = r'\n'.join(attrs) if attrs else ''
        method_str = r'\n'.join(methods) if methods else ''

        # Determine style based on inheritance
        style = 'filled'
        fillcolor = '#E8F4F8'
        if info['bases'] and info['bases'][0] not in ['Module', 'NamedTuple']:
            fillcolor = '#D0E8F2'

        # Create HTML-like label for record shape
        label = f'{{{class_name}|{attr_str}|{method_str}}}'

        dot.node(class_name, label, style=style, fillcolor=fillcolor, color='#4A90E2')

    # Draw inheritance edges
    for class_name, info in classes.items():
        for base in info['bases']:
            # Only show inheritance to custom classes, not built-ins
            if base in classes:
                dot.edge(class_name, base, style='solid', arrowhead='empty', color='#2E7D32', penwidth='2')
            elif base not in ['Module', 'NamedTuple', 'object']:
                # Show external inheritance
                dot.node(base, base, style='filled', fillcolor='#FFF9C4', color='#F57F17', shape='ellipse')
                dot.edge(class_name, base, style='dashed', arrowhead='empty', color='#F57F17')

    # Draw composition/usage relationships
    if relationships:
        for source, targets in relationships.items():
            for target in targets:
                if target in classes and source != target:
                    dot.edge(source, target, style='dashed', arrowhead='open', color='#E64A19', penwidth='1.5', label='uses')

    return dot


def main():
    src_dir = Path(__file__).parent / 'src'

    print(f"📊 Analyzing Python files in {src_dir}...")
    files = collect_python_files(str(src_dir))

    if not files:
        print(f"❌ No Python files found in {src_dir}")
        return

    print(f"✓ Found {len(files)} Python file(s)")
    for f in files:
        print(f"  - {f.relative_to(Path(__file__).parent)}")

    classes = analyze_python_files(files)

    if not classes:
        print("⚠️  No classes found in Python files")
        return

    print(f"\n✓ Found {len(classes)} class(es)")
    for class_name in sorted(classes.keys()):
        print(f"  - {class_name}")

    # Extract relationships between classes
    relationships = extract_relationships(files, classes)

    # Show detected relationships
    rel_count = sum(len(v) for v in relationships.values())
    if rel_count > 0:
        print(f"\n✓ Detected {rel_count} relationship(s)")
        for source, targets in sorted(relationships.items()):
            if targets:
                print(f"  {source} → {', '.join(sorted(targets))}")

    # Generate Graphviz diagram
    print("\n🎨 Generating UML diagram with Graphviz...")
    dot = generate_graphviz_diagram(classes, relationships)

    # Save and render
    output_file = Path(__file__).parent / 'uml_diagram'
    try:
        dot.render(str(output_file), cleanup=True)
        svg_file = output_file.with_suffix('.svg')
        print(f"✓ SVG diagram saved to {svg_file}")
        print(f"✓ View the diagram: {svg_file}")
    except Exception as e:
        print(f"❌ Error rendering diagram: {e}")
        print("   Make sure graphviz system package is installed:")
        print("   Ubuntu/Debian: sudo apt-get install graphviz")
        print("   macOS: brew install graphviz")


if __name__ == '__main__':
    main()