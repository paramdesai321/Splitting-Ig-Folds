use strict;
use warnings;

use lib '.';  # Ensure Perl can find the utils::MomOfInertiaAxes module
use utils::MomOfInertiaAxes;

# Sample input data for testing
 my $num_atoms = 3;
 my @masses = (1.0, 1.0, 1.0);
 my @x_coords = (1.0, 0.0, -1.0);
 my @y_coords = (0.0, 1.0, -1.0);
 my @z_coords = (0.0, 0.0, 0.0);
#
# # Call the function
 my ($xcom, $ycom, $zcom, $eigx, $eigy, $eigz) = MomOfInertiaAxes($num_atoms, \@masses, \@x_coords, \@y_coords, \@z_coords);

# Print output for verification
 print "Center of Mass: ($xcom, $ycom, $zcom)\n";
 print "Principal Axis with lowest inertia: ($eigx, $eigy, $eigz)\n";

