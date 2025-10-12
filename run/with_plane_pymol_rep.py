from draw_plane import draw_plane
import sys
PIN = sys.argv[0]
cmd.extend("draw_plane", draw_plane)
cmd.load(f"final_{PIN}_BCEF.pdb")
draw_plane(0,0,1,0)
cmd.save(f"final_{PIN}_BCEF_with_plane.pse")
