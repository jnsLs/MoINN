import torch
import matplotlib.pyplot as plt
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D


plt.ioff() # prevent figures from popping up

cpu = torch.device("cpu")
atom_names_dict = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 16: "S"}
atom_colors_dict = {1: "green", 6: "black", 7: "blue", 8: "red", 9: "purple"}


def vis_type_ass_on_molecule(mol, fig_name, node_colors):
    # define 2d embedding of molecule
    AllChem.Compute2DCoords(mol)
    mol = Draw.PrepareMolForDrawing(mol)

    # draw molecule and store as png
    d = rdMolDraw2D.MolDraw2DCairo(1000, 1000)
    d.drawOptions().fillHighlights = False
    d.DrawMoleculeWithHighlights(mol, '', node_colors, {}, {}, {}, -1)
    d.FinishDrawing()
    d.WriteDrawingText(fig_name + ".png")

    # draw molecule and store as svg
    d = rdMolDraw2D.MolDraw2DSVG(5000, 5000)
    d.drawOptions().fillHighlights = False
    d.DrawMoleculeWithHighlights(mol, '', node_colors, {}, {}, {}, -1)
    d.FinishDrawing()
    svg = d.GetDrawingText().replace('svg:', '')

    # store figure
    struc_form_pic = open(fig_name + ".svg", "w")
    struc_form_pic.write(svg)
    struc_form_pic.close()



def heatmap(fig, matrix):
    """
    plot heatmap of matrix.
    
    Args:
        matrix (2D torch Tensor): input Tensor 
    Returns:
        fig
    """
    # detach input Tensor and send them to cpu
    matrix = matrix.to(cpu).detach().numpy()
    # visualize    
    ##fig = plt.figure()
    ##fig.add_subplot(1,1,1)
    plt.imshow(matrix)
    plt.colorbar()
    ##fig.canvas.draw()
    return fig
        



