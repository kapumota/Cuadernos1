def dibuja_grafo_unica_capa_oculta():
    import graphviz
    grafo_nn = graphviz.Digraph(node_attr={'shape': 'circle', 'fixedsize': 'True'},
                                graph_attr={'rankdir': 'LR', 'splines': 'line'})

    entradas = graphviz.Digraph(node_attr={'shape': 'circle'}, name="cluster_0")
    capaOculta = graphviz.Digraph(node_attr={'shape': 'circle'}, name="cluster_1")
    salida = graphviz.Digraph(node_attr={'shape': 'circle'}, name="cluster_2")

    for i in range(4):
        entradas.node("x[%d]" % i)

    entradas.body.append('label = "entradas"')
    entradas.body.append('color = "green"')

    capaOculta.body.append('label = "capa oculta"')
    capaOculta.body.append('color = "red"')

    for i in range(3):
        capaOculta.node("h%d" % i, label="h[%d]" % i)

    salida.node("y")
    salida.body.append('label = "salida"')
    salida.body.append('color = "blue"')

    grafo_nn.subgraph(entradas)
    grafo_nn.subgraph(capaOculta)
    grafo_nn.subgraph(salida)

    for i in range(4):
        for j in range(3):
            grafo_nn.edge("x[%d]" % i, "h%d" % j)

    for i in range(3):
        grafo_nn.edge("h%d" % i, "y")
    return grafo_nn
