from graphviz import Digraph

# Create MLP graph
mlp = Digraph()
mlp.attr(rankdir='LR', label='Single Hidden Layer MLP', labelloc='t')

# Layers
mlp.node('input', 'Input Layer\n(input_dim)', shape='box')
mlp.node('hidden', 'Hidden Layer\n(hidden_dim)', shape='box')
mlp.node('output', 'Output Layer\n(output_dim)', shape='box')

# Connections
mlp.edges([('input', 'hidden'), ('hidden', 'output')])

# Render
mlp.render('mlp.gv', view=True)