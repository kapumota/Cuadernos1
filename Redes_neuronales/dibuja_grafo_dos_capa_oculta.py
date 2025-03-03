def dibuja_grafo_dos_capa_oculta():
    import graphviz
    grafo_nn = graphviz.Digraph(node_attr={'shape': 'circle', 'fixedsize': 'True'},
                                graph_attr={'rankdir': 'LR', 'splines': 'line'})

    entradas = graphviz.Digraph(node_attr={'shape': 'circle'}, name="cluster_0")
    capaOculta = graphviz.Digraph(node_attr={'shape': 'circle'}, name="cluster_1")
    capaOculta1 = graphviz.Digraph(node_attr={'shape': 'circle'}, name="cluster_2")
    
    salida = graphviz.Digraph(node_attr={'shape': 'circle'}, name="cluster_3")

    for i in range(4):
        entradas.node("x[%d]" % i)

    entradas.body.append('label = "entradas"')
    entradas.body.append('color = "green"')

    for i in range(3):
        capaOculta.node("h1[%d]" % i)
        
    for i in range(3):
        capaOculta1.node("h2[%d]" % i)
    
    capaOculta.body.append('label = "Capa oculta 1"')
    capaOculta.body.append('color = "orange"')
    
    capaOculta1.body.append('label = "Capa oculta 2"')
    capaOculta1.body.append('color = "yellow"')
        
        

    salida.node('y')
    salida.body.append('label = "salida"')
    salida.body.append('color = "blue"')

    grafo_nn.subgraph(entradas)
    grafo_nn.subgraph(capaOculta)
    grafo_nn.subgraph(capaOculta1)
    
    grafo_nn.subgraph(salida)

    for i in range(4):
        for j in range(3):
            grafo_nn.edge("x[%d]" % i, "h1[%d]" % j, label="")

    for i in range(3):
        for j in range(3):
            grafo_nn.edge("h1[%d]" % i, "h2[%d]" % j, label="")

    for i in range(3):
        grafo_nn.edge("h2[%d]" % i, "y", label="")

    return grafo_nn
