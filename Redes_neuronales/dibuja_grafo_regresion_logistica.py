def dibuja_grafo_regresion_logistica():
    import graphviz
    grafo_rl = graphviz.Digraph(node_attr={'shape': 'circle', 'fixedsize': 'True'},
                                graph_attr={'rankdir': 'LR', 'splines': 'line'})
    entrada = graphviz.Digraph(node_attr={'shape': 'circle'}, name="cluster_0")
    salida = graphviz.Digraph(node_attr={'shape': 'circle'}, name="cluster_2")

    for i in range(4):
        entrada.node("x[%d]" % i, labelloc="c")
    entrada.body.append('label = "entrada"')
    entrada.body.append('color = "green"')

    grafo_rl.subgraph(entrada)

    salida.body.append('label = "salida"')
    salida.body.append('color = "red"')
    salida.node("y")

    grafo_rl.subgraph(salida)

    for i in range(4):
        grafo_rl.edge("x[%d]" % i, "y", label="w[%d]" % i)
    return grafo_rl
