# @author: Rafael C. Bernardi

draw material Glossy
color Display Background white
display projection Perspective
axes location Off
display resize 1280 720

for {set i 0} {$i < $numstep} {incr i} {	 
mol top $i

mol modselect 0 top "protein"
mol modcolor 0 top SegName
mol modstyle 0 top NewCartoon 0.30000 20.00000 4.10000 0
mol modmaterial 0 top AOChalky

mol addrep top
mol modselect 1 top "nucleic"
mol modcolor 1 top SegName
mol modstyle 1 top NewCartoon 0.30000 20.00000 4.10000 0
mol modmaterial 1 top AOChalky

mol addrep top
mol modselect 2 top "lipid"
mol modcolor 2 top Name
mol modstyle 2 top Licorice 0.30000 20.00000 20.00000
mol modmaterial 2 top AOChalky

mol addrep top
mol modselect 3 top "sugar"
mol modcolor 3 top Name
mol modstyle 3 top Licorice 0.30000 20.00000 20.00000
mol modmaterial 3 top AOChalky

mol addrep top
mol modselect 4 top "water"
mol modcolor 4 top Name
mol modstyle 4 top VDW 1.00000 20.000000
mol modmaterial 4 top AOChalky

mol addrep top
mol modselect 5 top "ions"
mol modcolor 5 top Name
mol modstyle 5 top VDW 1.00000 20.000000
mol modmaterial 5 top AOChalky

mol addrep top
mol modselect 6 top "resname $ligand"
mol modcolor 6 top Name
mol modstyle 6 top CPK 1.00000 0.300000 20.000000 20.000000
mol modmaterial 6 top AOEdgy

}


material change outline AOEdgy 2.0000
material change outlinewidth AOEdgy 0.9500

material change outline AOChalky 2.0000
material change outlinewidth AOChalky 0.5500

material change outline Glossy 2.0000
material change outlinewidth Glossy 0.5500

color Name C gray
display nearclip set 0.010000
display height 1.500000
display cuedensity 0.150000
display dof on
display dof_fnumber 300.00000
display dof_focaldist 1.850000
display shadows on
display ambientocclusion on

colorscheme Network

render aosamples TachyonLOptiXInternal 1
render aasamples TachyonLOptiXInternal 512

mol top 0


