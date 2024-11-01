from pymol import cmd, cgo

# Define the vertices and colors
vertices = [
    cgo.COLOR, 0.0, 0.0, 1.0,  # Color (blue)
    cgo.ALPHA, 0.5,            # Transparency
    cgo.BEGIN, cgo.QUADS,
    cgo.VERTEX, 1.0, 1.0, 0.0,
    cgo.VERTEX, 1.0, -1.0, 0.0,
    cgo.VERTEX, -1.0, -1.0, 0.0,
    cgo.VERTEX, -1.0, 1.0, 0.0,
    cgo.END
]

# Load the CGO object
cmd.load_cgo(vertices, "plane")

# Show the plane
cmd.show("cgo", "plane")
