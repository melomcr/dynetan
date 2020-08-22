# @author: Rafael C. Bernardi

proc networkedges {stdcolor window resolution basename numstep} {
 for {set i 0} {$i < $numstep} {incr i} {	 
 mol top $i
  # to start delete other network information
  draw delete all
  # 
  set whatload _AllEdges_window_
  set infile [open $basename$whatload$i.dat r]
      while {[gets $infile line] >=0} {
        # reads 1st column
        set atom1 [expr [lindex $line 0]]
        # reads 2nd column
        set atom2 [expr [lindex $line 1]]
        # reads 3rd column
        set weight [lindex $line 2]
        # get selections
        set sel1 [atomselect top "index $atom1"]
        set sel2 [atomselect top "index $atom2"]  
        # draws the edges
        draw color $stdcolor
        draw cylinder [join [$sel1 get {x y z}]] [join [$sel2 get {x y z}]] radius $weight resolution $resolution
        draw sphere [join [$sel1 get {x y z}]] radius $weight resolution $resolution
        draw sphere [join [$sel2 get {x y z}]] radius $weight resolution $resolution
       }
 }
 mol top $window
}

proc networknodes {stdcolor window resolution basename numstep} {
 for {set i 0} {$i < $numstep} {incr i} {	 
 mol top $i
  # to start delete other network information
  draw delete all
  # 
  set whatload _AllNodes_window_
  set infile [open $basename$whatload$i.dat r]
      while {[gets $infile line] >=0} {
        # reads 1th column
        set atom1 [expr [lindex $line 0]]
        # reads 2th column
        set weight [lindex $line 1]
        # get selections
        set sel1 [atomselect top "index $atom1"]
        # draws the nodes
        draw color $stdcolor
        draw sphere [join [$sel1 get {x y z}]] radius $weight resolution $resolution
      }
 }
 mol top $window
}

proc communityedges {window resolution basename numstep} {
 for {set i 0} {$i < $numstep} {incr i} {	 
 mol top $i
  # to start delete other network information
  draw delete all
  #  
  set whatload _IntraCommunities_window_
  set infile [open $basename$whatload$i.dat r]
      while {[gets $infile line] >=0} {
        # reads 1st column
        set atom1 [expr [lindex $line 0]]
        # reads 2nd column
        set atom2 [expr [lindex $line 1]]
        # reads 3rd column
        set weight [lindex $line 2]
        # reads 4th column
        set color [lindex $line 3]
        # get selections
        set sel1 [atomselect top "index $atom1"]
        set sel2 [atomselect top "index $atom2"]  
        # draws the edges
        draw color [expr $color + 40]
        draw cylinder [join [$sel1 get {x y z}]] [join [$sel2 get {x y z}]] radius $weight resolution $resolution
        draw sphere [join [$sel1 get {x y z}]] radius $weight resolution $resolution
        draw sphere [join [$sel2 get {x y z}]] radius $weight resolution $resolution
       }
 }
 mol top $window
}

proc communitynodes {window resolution basename numstep} {
 for {set i 0} {$i < $numstep} {incr i} {	 
 mol top $i
  # to start delete other network information
  draw delete all
  # 
  set whatload _AllNodes_window_
  set infile [open $basename$whatload$i.dat r]
      while {[gets $infile line] >=0} {
        # reads 1th column
        set atom1 [expr [lindex $line 0]]
        # reads 2th column
        set weight [lindex $line 1]
        # reads 3rd column
        set color [lindex $line 2]
        # get selections
        set sel1 [atomselect top "index $atom1"]
        # draws the nodes
        draw color [expr $color + 40]
        draw sphere [join [$sel1 get {x y z}]] radius $weight resolution $resolution
      }
 }
 mol top $window
}

proc intercommunities {intercolor window resolution basename numstep} {
 for {set i 0} {$i < $numstep} {incr i} {	 
 mol top $i
  # to start delete other network information
  #draw delete all
  # 
  set whatload _InterCommunities_window_
  set infile [open $basename$whatload$i.dat r]
      while {[gets $infile line] >=0} {
        # reads 1st column
        set atom1 [expr [lindex $line 0]]
        # reads 2nd column
        set atom2 [expr [lindex $line 1]]
        # reads 3rd column
        set weight [lindex $line 2]
        # get selections
        set sel1 [atomselect top "index $atom1"]
        set sel2 [atomselect top "index $atom2"]  
        # define width of the lines
        if {$weight <= 0.25} {
        set width 1
        } elseif {$weight <= 0.5} {
        set width 2
        } elseif {$weight <= 0.75} {
        set width 3
        } else {
        set width 4
        }
        # draws the edges
        draw color $intercolor
        draw line [join [$sel1 get {x y z}]] [join [$sel2 get {x y z}]] width $width style dashed
#        draw cylinder [join [$sel1 get {x y z}]] [join [$sel2 get {x y z}]] radius $weight resolution $resolution
       }
 }
 mol top $window
}

proc betweennesscorr {stdcolor window resolution basename numstep} {
 for {set i 0} {$i < $numstep} {incr i} {	 
 mol top $i
  # to start delete other network information
  draw delete all
  # 
  set whatload _Betweenness_window_
  set infile [open $basename$whatload$i.dat r]
      while {[gets $infile line] >=0} {
        # reads 1st column
        set atom1 [expr [lindex $line 0]]
        # reads 2nd column
        set atom2 [expr [lindex $line 1]]
        # reads 3rd column
        set weight [lindex $line 2]
        # reads 4th column
#        set weight [lindex $line 3]
        # get selections
        set sel1 [atomselect top "index $atom1"]
        set sel2 [atomselect top "index $atom2"]  
        # draws the edges
        draw color $stdcolor
        draw cylinder [join [$sel1 get {x y z}]] [join [$sel2 get {x y z}]] radius $weight resolution $resolution
        draw sphere [join [$sel1 get {x y z}]] radius $weight resolution $resolution
        draw sphere [join [$sel2 get {x y z}]] radius $weight resolution $resolution
       }
 }
 mol top $window
}

proc betweenness {stdcolor window resolution basename numstep} {
 for {set i 0} {$i < $numstep} {incr i} {	 
 mol top $i
  # to start delete other network information
  draw delete all
  # 
  set whatload _Betweenness_window_
  set infile [open $basename$whatload$i.dat r]
      while {[gets $infile line] >=0} {
        # reads 1st column
        set atom1 [expr [lindex $line 0]]
        # reads 2nd column
        set atom2 [expr [lindex $line 1]]
        # reads 3rd column
#        set weight [lindex $line 2]
        # reads 4th column
        set weight [lindex $line 3]
        # get selections
        set sel1 [atomselect top "index $atom1"]
        set sel2 [atomselect top "index $atom2"]  
        # draws the edges
        draw color $stdcolor
        draw cylinder [join [$sel1 get {x y z}]] [join [$sel2 get {x y z}]] radius $weight resolution $resolution
        draw sphere [join [$sel1 get {x y z}]] radius $weight resolution $resolution
        draw sphere [join [$sel2 get {x y z}]] radius $weight resolution $resolution
       }
 }
 mol top $window
}

proc paths {mode rep stdcolor window resolution basename numstep path} {
 for {set i 0} {$i < $numstep} {incr i} {	 
 mol top $i
  # to start delete other network information
  draw delete all
  # 
  set whatload _Paths_window_
  set infile [open $basename$whatload$i$path.dat r]
    while {[gets $infile line] >=0} {
      # test if optimal or suboptimal
      # reads 6th column
      set tmp [lindex $line 5]
      if {$tmp >= 1} {
        # reads 1st column
        set atom1 [expr [lindex $line 0]]
        # reads 2nd column
        set atom2 [expr [lindex $line 1]]
          # get mode >>> correlation or betweeness
          if {$mode == "correlation"} {
          # reads 3rd column
          set weight [lindex $line 2]
          } else {
          # reads 5th column
          set weight [lindex $line 4]
          }
        # reads 6th column
        set tmp_color [lindex $line 5]
        # get selections
        set sel1 [atomselect top "index $atom1"]
        set sel2 [atomselect top "index $atom2"]  
        # draws the edges
          if {$rep == "dashed"} {
             if {$tmp_color == 0} {
             set color 36
             draw color $color
             draw cylinder [join [$sel1 get {x y z}]] [join [$sel2 get {x y z}]] radius $weight resolution $resolution
             draw sphere [join [$sel1 get {x y z}]] radius $weight resolution $resolution
             draw sphere [join [$sel2 get {x y z}]] radius $weight resolution $resolution
             } else {
             set color 33
             draw color $color
             draw line [join [$sel1 get {x y z}]] [join [$sel2 get {x y z}]] width 3 style dashed
             draw sphere [join [$sel1 get {x y z}]] radius 0.3 resolution $resolution
             draw sphere [join [$sel2 get {x y z}]] radius 0.3 resolution $resolution
#             draw sphere [join [$sel1 get {x y z}]] radius $weight resolution $resolution
#             draw sphere [join [$sel2 get {x y z}]] radius $weight resolution $resolution
             }
          } elseif {$rep == "cylinder"} {
             if {$tmp_color == 0} {
             set color 36
             draw color $color
             draw cylinder [join [$sel1 get {x y z}]] [join [$sel2 get {x y z}]] radius $weight resolution $resolution
             draw sphere [join [$sel1 get {x y z}]] radius $weight resolution $resolution
             draw sphere [join [$sel2 get {x y z}]] radius $weight resolution $resolution
             } else {
             set color 33
             draw color $color
             draw cylinder [join [$sel1 get {x y z}]] [join [$sel2 get {x y z}]] radius $weight resolution $resolution
             draw sphere [join [$sel1 get {x y z}]] radius $weight resolution $resolution
             draw sphere [join [$sel2 get {x y z}]] radius $weight resolution $resolution
             }
          } elseif {$rep == "optimal"} {
          } else { 
             set color $stdcolor
             draw color $color
             draw cylinder [join [$sel1 get {x y z}]] [join [$sel2 get {x y z}]] radius $weight resolution $resolution
             draw sphere [join [$sel1 get {x y z}]] radius $weight resolution $resolution
             draw sphere [join [$sel2 get {x y z}]] radius $weight resolution $resolution
          }
        }
     }
  set infile [open $basename$whatload$i$path.dat r]
    while {[gets $infile line] >=0} {
      # test if optimal or suboptimal
      # reads 6th column
      set tmp [lindex $line 5]
      if {$tmp == 0} {
        # reads 1st column
        set atom1 [expr [lindex $line 0]]
        # reads 2nd column
        set atom2 [expr [lindex $line 1]]
          # get mode >>> correlation or betweeness
          if {$mode == "correlation"} {
          # reads 3rd column
          set weight [lindex $line 2]
          } else {
          # reads 5th column
          set weight [lindex $line 4]
          }
        # reads 6th column
        set tmp_color [lindex $line 5]
        # get selections
        set sel1 [atomselect top "index $atom1"]
        set sel2 [atomselect top "index $atom2"]  
        # draws the edges
          if {$rep == "dashed"} {
             if {$tmp_color == 0} {
             set color 36
             draw color $color
             draw cylinder [join [$sel1 get {x y z}]] [join [$sel2 get {x y z}]] radius $weight resolution $resolution
             draw sphere [join [$sel1 get {x y z}]] radius $weight resolution $resolution
             draw sphere [join [$sel2 get {x y z}]] radius $weight resolution $resolution
             } else {
             set color 33
             draw color $color
             draw line [join [$sel1 get {x y z}]] [join [$sel2 get {x y z}]] width 3 style dashed
             draw sphere [join [$sel1 get {x y z}]] radius 0.3 resolution $resolution
             draw sphere [join [$sel2 get {x y z}]] radius 0.3 resolution $resolution
#             draw sphere [join [$sel1 get {x y z}]] radius $weight resolution $resolution
#             draw sphere [join [$sel2 get {x y z}]] radius $weight resolution $resolution
             }
          } elseif {$rep == "cylinder"} {
             if {$tmp_color == 0} {
             set color 36
             draw color $color
             draw cylinder [join [$sel1 get {x y z}]] [join [$sel2 get {x y z}]] radius $weight resolution $resolution
             draw sphere [join [$sel1 get {x y z}]] radius $weight resolution $resolution
             draw sphere [join [$sel2 get {x y z}]] radius $weight resolution $resolution
             } else {
             set color 33
             draw color $color
             draw cylinder [join [$sel1 get {x y z}]] [join [$sel2 get {x y z}]] radius $weight resolution $resolution
             draw sphere [join [$sel1 get {x y z}]] radius $weight resolution $resolution
             draw sphere [join [$sel2 get {x y z}]] radius $weight resolution $resolution
             }
          } else {
             set color $stdcolor
             draw color $color
             draw cylinder [join [$sel1 get {x y z}]] [join [$sel2 get {x y z}]] radius $weight resolution $resolution
             draw sphere [join [$sel1 get {x y z}]] radius $weight resolution $resolution
             draw sphere [join [$sel2 get {x y z}]] radius $weight resolution $resolution
          }
        }
     }
 }
 mol top $window
}



