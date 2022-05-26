# noinspection PyUnresolvedReferences

from fixtures import panda_fsm as Panda


def test_nx_export(Panda):
    subject = Panda("Pekka")
    G = subject.make_networkx_graph()
    import matplotlib.pyplot as plt
    import networkx as nx

    pos = nx.spring_layout(G)
    #node_labels = nx.get_node_attributes(G, 'state')
    edge_labels = nx.get_edge_attributes(G, "label")
    nx.draw_networkx(G, pos, )
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.show()


def test_pydot_export(Panda):
    from PIL import Image
    from io import BytesIO
    subject = Panda("Pekka")
    G = subject.make_dot_graph()

    print(G.to_string())
    Image.open(BytesIO(G.create_png())).show()


