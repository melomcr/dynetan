############################
#
# To run this script execute the command: vmd -e network_view_2.tcl
#
# @author: Rafael C. Bernardi
#
############################

# System information

set basename BASENAMETMP
set numstep  NUMSTEPTMP
set ligand   LIGANDTMP


#############################
# User defined preferences: #
#############################

# Special bright colors:
# 33 > Red
# 34 > Yellow
# 35 > Green
# 36 > Blue
# 37 > Pink
# 38 > Violet
# 39 > Brown

set stdcolor 33
set intercolor 37

##########################
# Do not change ##########
##########################

# Load all molecules

set whatload _Structure_window_
for {set i 0} {$i < $numstep} {incr i} {	 
mol new $basename$whatload$i.pdb
}


# Set default Resolution
set resolution 12


# Load all the modules
source PATHTMP/network_menu.tcl
source PATHTMP/network_proc.tcl
source PATHTMP/network_color.tcl
source PATHTMP/network_rep.tcl

# Ste loading

set buttonprot on

# Load Network View 2.0 menu
drawControlWindow $stdcolor $intercolor $resolution $basename $numstep

# Showing window 0 first
mol off all
set window 0
mol top $window
mol on $window

# Preselect Path
set infile [open paths.list r]
 set idx 0
 while {[gets $infile line] >=0} {
 set idx [expr $idx + 1]
  if {$idx == 1} {
  set linetext [lindex $line 0]
  set path $linetext
  }
 }


