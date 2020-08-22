# @author: Rafael C. Bernardi

set buttonprot on
set buttonnucleic on
set buttonlipid on
set buttonsugar on
set buttonwater on
set buttonions on
set buttonligand on
set buttontransp off
set buttoncommres off

proc drawControlWindow {stdcolor intercolor resolution basename numstep} {

    #This is the width of the buttons - make it larger if you have longer button names
    set bwidth 40
    set bwidthsmall 30


    #Create a new tk window
    set w [toplevel .network]
    wm title $w "Network View 2.0"
    
    #Define the buttons
    labelframe $w.step -bd 2 -relief ridge -text "Steps"


for {set i 0} {$i < $numstep} {incr i} {	 
    set step [expr $i + 1] 
    button $w.step.show$step -text "Step $step" -command [subst {
          mol off all
          set window $i
          mol top \$window
          mol on \$window 
      } ]  -width $bwidthsmall

}



    #Define the buttons
    labelframe $w.repS -bd 2 -relief ridge -text "Representations"

    #create button command
    button $w.repS.show1 -text "Show/Hide Protein" -command {
      #the if/else tests to see if representation is on or off.
      if {$buttonprot == "on"} {
          for {set i 0} {$i < $numstep} {incr i} {mol showrep $i 0 0}
          set buttonprot off
      } else {
     	  for {set i 0} {$i < $numstep} {incr i} {mol showrep $i 0 1}
          set buttonprot on
      }
    }  -width $bwidth

    button $w.repS.show2 -text "Show/Hide Nucleic" -command {
      #the if/else tests to see if representation is on or off.
      if {$buttonnucleic == "on"} {
          for {set i 0} {$i < $numstep} {incr i} {mol showrep $i 1 0}
          set buttonnucleic off
      } else {
     	  for {set i 0} {$i < $numstep} {incr i} {mol showrep $i 1 1}
          set buttonnucleic on
      }
    }  -width $bwidth

    button $w.repS.show3 -text "Show/Hide Lipid" -command {
      #the if/else tests to see if representation is on or off.
      if {$buttonlipid == "on"} {
          for {set i 0} {$i < $numstep} {incr i} {mol showrep $i 2 0}
          set buttonlipid off
      } else {
     	  for {set i 0} {$i < $numstep} {incr i} {mol showrep $i 2 1}
          set buttonlipid on
      }
    }  -width $bwidth

    button $w.repS.show4 -text "Show/Hide Sugar" -command {
      #the if/else tests to see if representation is on or off.
      if {$buttonsugar == "on"} {
          for {set i 0} {$i < $numstep} {incr i} {mol showrep $i 3 0}
          set buttonsugar off
      } else {
     	  for {set i 0} {$i < $numstep} {incr i} {mol showrep $i 3 1}
          set buttonsugar on
      }
    }  -width $bwidth

    button $w.repS.show5 -text "Show/Hide Water" -command {
      #the if/else tests to see if representation is on or off.
      if {$buttonwater == "on"} {
          for {set i 0} {$i < $numstep} {incr i} {mol showrep $i 4 0}
          set buttonwater off
      } else {
     	  for {set i 0} {$i < $numstep} {incr i} {mol showrep $i 4 1}
          set buttonwater on
      }
    }  -width $bwidth

    button $w.repS.show6 -text "Show/Hide Ions" -command {
      #the if/else tests to see if representation is on or off.
      if {$buttonions == "on"} {
          for {set i 0} {$i < $numstep} {incr i} {mol showrep $i 5 0}
          set buttonions off
      } else {
     	  for {set i 0} {$i < $numstep} {incr i} {mol showrep $i 5 1}
          set buttonions on
      }
    }  -width $bwidth

    button $w.repS.show7 -text "Show/Hide Ligand" -command {
      #the if/else tests to see if representation is on or off.
      if {$buttonligand == "on"} {
          for {set i 0} {$i < $numstep} {incr i} {mol showrep $i 6 0}
          set buttonligand off
      } else {
     	  for {set i 0} {$i < $numstep} {incr i} {mol showrep $i 6 1}
          set buttonligand on
      }
    }  -width $bwidth






    labelframe $w.rep -bd 2 -relief ridge -text "Structure Representation"

    button $w.rep.show1 -text "Transparent/Opaque" -command {
      #the if/else tests to see if transparency is on or off.
      if {$buttontransp == "off"} {
          material change opacity AOChalky 0.300000
          set buttontransp on
      } else {
     	  material change opacity AOChalky 1.000000
          set buttontransp off
      }
    }  -width $bwidth





    labelframe $w.render -bd 2 -relief ridge -text "Render"

    button $w.render.show1 -text "Image Rendering (GPU) - Interactive" -command {
      display dof on
 	  render TachyonLOptiXInteractive network.ppm display %s
      return
      }  -width $bwidth

    button $w.render.show2 -text "Image Rendering (GPU)" -command {
      display dof off
	  render TachyonLOptiXInternal network.ppm display %s
      }  -width $bwidth

    button $w.render.show3 -text "Image Rendering" -command {
      display dof off
	  render TachyonInternal vmdscene.tga display %s
      }  -width $bwidth






    labelframe $w.net -bd 2 -relief ridge -text "Network Representations"

    button $w.net.show1 -text "Full Network" -command {
	       networkedges $stdcolor $window $resolution $basename $numstep 
     }  -width $bwidthsmall


     button $w.net.show2 -text "All Nodes" -command {
           networknodes $stdcolor $window $resolution $basename $numstep
     }  -width $bwidthsmall

    button $w.net.show3 -text "All Communities" -command {
	       communityedges $window $resolution $basename $numstep
     }  -width $bwidthsmall

    button $w.net.show4 -text "All Community Nodes" -command {
           communitynodes $window $resolution $basename $numstep
     }  -width $bwidthsmall 

    button $w.net.show5 -text "Inter-Community Links" -command {
           intercommunities $intercolor $window $resolution $basename $numstep
     }  -width $bwidthsmall

#    button $w.net.show6 -text "Betweeness (by gen. corr.)" -command {
#           betweennesscorr $stdcolor $window $resolution $basename $numstep
#     }  -width $bwidthsmall

    button $w.net.show7 -text "Betweeness" -command {
           betweenness $stdcolor $window $resolution $basename $numstep
     }  -width $bwidthsmall

    button $w.net.show8 -text "Hide Network" -command {draw delete all} -width $bwidthsmall



    labelframe $w.r -bd 2 -relief ridge -text "Select a Path"
    set infile [open paths.list r]
    set idx 0
      while {[gets $infile line] >=0} {
      set idx [expr $idx + 1]
      set linetext [lindex $line 0]
      radiobutton $w.r.$idx -text "Path $linetext" -variable path -value $linetext -width $bwidth
      }

    labelframe $w.netpath -bd 2 -relief ridge -text "Pathways (weighted by gen. correlation)"

    button $w.netpath.show1 -text "Suboptimal Paths - highlight optimal" -command {
           paths correlation cylinder $stdcolor $window $resolution $basename $numstep $path
     }  -width $bwidth

    button $w.netpath.show2 -text "Suboptimal Paths" -command {
           paths correlation monochromo $stdcolor $window $resolution $basename $numstep $path
     }  -width $bwidth

    button $w.netpath.show3 -text "Optimal Path with dashed suboptimal" -command {
           paths correlation dashed $stdcolor $window $resolution $basename $numstep $path
     }  -width $bwidth

    button $w.netpath.show4 -text "Optimal Path" -command {
           paths correlation optimal $stdcolor $window $resolution $basename $numstep $path
     }  -width $bwidth




    labelframe $w.netpathbet -bd 2 -relief ridge -text "Pathways (weighted by betweeness)"

    button $w.netpathbet.show1 -text "Suboptimal Paths - highlight optimal" -command {
           paths betweeness cylinder $stdcolor $window $resolution $basename $numstep $path
     }  -width $bwidth

    button $w.netpathbet.show2 -text "Suboptimal Paths" -command {
           paths betweeness monochromo $stdcolor $window $resolution $basename $numstep $path
     }  -width $bwidth

    button $w.netpathbet.show3 -text "Optimal Path with dashed suboptimal" -command {
           paths betweeness dashed $stdcolor $window $resolution $basename $numstep $path
     }  -width $bwidth

    button $w.netpathbet.show4 -text "Optimal Path" -command {
           paths betweeness optimal $stdcolor $window $resolution $basename $numstep $path
     }  -width $bwidth

    


    labelframe $w.selres -bd 2 -relief ridge -text "Network Drawing Resolution (Nodes & Edges)"
    radiobutton $w.selres.show1 -text "Low Resolution" -variable resolution -value 3 -width $bwidth
    radiobutton $w.selres.show2 -text "Normal Resolution" -variable resolution -value 6 -width $bwidth
    radiobutton $w.selres.show3 -text "High Resolution" -variable resolution -value 12 -width $bwidth
    radiobutton $w.selres.show4 -text "Very High Resolution" -variable resolution -value 20 -width $bwidth


    labelframe $w.commres -bd 2 -relief ridge -text "Color Protein by Communities"


    button $w.commres.show1 -text "Show/Hide Colors" -command {
    set maxvalue 0
    set number 0

      #the if/else tests to see if representation is on or off.
      if {$buttoncommres == "off"} {

          for {set j 0} {$j < $numstep} {incr j} {	 
          mol top $j

           for {set k 0} {$k <= 6} {incr k} {
             mol showrep $j $k 0
           }


            [atomselect top all] set beta 0.0
            set whatload _AllNodes_window_
            set infile [open $basename$whatload$j.dat r]
               while {[gets $infile line] >=0} {
               set atom [lindex $line 0]
               set value [lindex $line 2]
               [atomselect top "protein and same residue as index $atom"] set beta $value
                 if {$value > $maxvalue} {
                 set maxvalue $value
                 }
               }
            for {set i 0} {$i <= $maxvalue} {incr i} {
            set number [expr $i + 7]
            mol addrep top
            mol modselect $number top "protein and beta $i"
            mol modcolor $number top ColorID $i
            mol modstyle $number top NewCartoon 0.30000 20.00000 4.10000 0
            mol modmaterial $number top AOChalky
            }
          }
          mol top $window

        set buttoncommres on
      } else {

           for {set j 0} {$j < $numstep} {incr j} {	 
           mol top $j

            set loadfile _AllNodes_window_
            set closefile [open $basename$loadfile$j.dat r]
               while {[gets $closefile line] >=0} {
               set value [lindex $line 2]
                 if {$value > $maxvalue} {
                 set maxvalue $value
                 }
               }
            for {set i 0} {$i <= $maxvalue} {incr i} {
            set number [expr $i + 7]
              for {set k 7} {$k <= $number} {incr k} {
              mol delrep $k $j   
              }
            }
           }


           if {$buttonprot == "on"} {
               for {set i 0} {$i < $numstep} {incr i} {mol showrep $i 0 1}
           }
           if {$buttonnucleic == "on"} {
               for {set i 0} {$i < $numstep} {incr i} {mol showrep $i 1 1}
           }
           if {$buttonlipid == "on"} {
               for {set i 0} {$i < $numstep} {incr i} {mol showrep $i 2 1}
           }
           if {$buttonsugar == "on"} {
               for {set i 0} {$i < $numstep} {incr i} {mol showrep $i 3 1}
           }
           if {$buttonwater == "on"} {
               for {set i 0} {$i < $numstep} {incr i} {mol showrep $i 4 1}
           }
           if {$buttonions == "on"} {
               for {set i 0} {$i < $numstep} {incr i} {mol showrep $i 5 1}
           }   
           if {$buttonligand == "on"} {
               for {set i 0} {$i < $numstep} {incr i} {mol showrep $i 6 1}
           }
           mol top $window

        set buttoncommres off
      }
    }  -width $bwidthsmall








    #Add the buttons to the labelframe
for {set i 1} {$i <= $numstep} {incr i} {	 
    pack $w.step.show$i -fill x
}

    pack $w.repS.show1 -fill x
    pack $w.repS.show2 -fill x
    pack $w.repS.show3 -fill x
    pack $w.repS.show4 -fill x
    pack $w.repS.show5 -fill x
    pack $w.repS.show6 -fill x
    pack $w.repS.show7 -fill x

    pack $w.rep.show1 -fill x

    pack $w.render.show1 -fill x
    pack $w.render.show2 -fill x
    pack $w.render.show3 -fill x

    pack $w.net.show1 -fill x
    pack $w.net.show2 -fill x
    pack $w.net.show3 -fill x
    pack $w.net.show4 -fill x
    pack $w.net.show5 -fill x
#    pack $w.net.show6 -fill x
    pack $w.net.show7 -fill x
    pack $w.net.show8 -fill x

    pack $w.netpath.show1 -fill x
    pack $w.netpath.show2 -fill x
    pack $w.netpath.show3 -fill x
    pack $w.netpath.show4 -fill x

    pack $w.netpathbet.show1 -fill x
    pack $w.netpathbet.show2 -fill x
    pack $w.netpathbet.show3 -fill x
    pack $w.netpathbet.show4 -fill x

    pack $w.selres.show1 -fill x
    pack $w.selres.show2 -fill x
    pack $w.selres.show3 -fill x
    pack $w.selres.show4 -fill x

    pack $w.commres.show1 -fill x
    #pack $w.commres.show2 -fill x
     
for {set i 1} {$i <= $idx} {incr i} {	 
    pack $w.r.$i -fill x

}


#pack $w.l.show -fill x

    #Add the labelframe to the window
    grid $w.step        -padx 0 -columnspan 1 -column 0 -row 0  -rowspan 6 -sticky ns
    grid $w.net         -padx 0 -columnspan 2 -column 0 -row 6  -rowspan 7 -sticky ns
    grid $w.commres     -padx 0 -columnspan 2 -column 0 -row 13 -rowspan 2 -sticky ns 

    grid $w.r           -padx 0 -columnspan 1 -column 2 -row 0  -rowspan 7 -sticky w
    grid $w.netpath     -padx 0 -columnspan 1 -column 2 -row 7  -rowspan 4 -sticky ns 
    grid $w.netpathbet  -padx 0 -columnspan 1 -column 2 -row 11 -rowspan 4 -sticky ns  



    grid $w.rep         -padx 0 -columnspan 1 -column 3 -row 0  -rowspan 1 -sticky ns 
    grid $w.selres      -padx 0 -columnspan 1 -column 3 -row 1  -rowspan 4 -sticky ns
    grid $w.repS        -padx 0 -columnspan 1 -column 3 -row 5  -rowspan 7 -sticky ns 
    grid $w.render      -padx 0 -columnspan 1 -column 3 -row 12 -rowspan 3 -sticky ns





#grid $w.l -row 0 -column 3 -rowspan 1 -sticky news
#grid $w.s -row 0 -column 4 -rowspan 1 -sticky news
#grid $w.label -row 1 -column 3 -columnspan 2


}

