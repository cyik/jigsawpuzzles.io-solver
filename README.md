#Jigsawpuzzles.io solver by cyik

An automated jigsaw puzzle solver for jigsawpuzzles.io that tells you where every piece of a puzzle fits on a puzzle board. Simply screenshot the puzzle piece using snipping tool (Win + Shift + S) and the solver7.py scans the piece and overlays a green box directly on your screen at the exact spot where it belongs. Read the README word file to see setup instructions, you must follow it or it won't work.

![image alt](https://github.com/cyik/jigsawpuzzles.io-solver/blob/969105b1a7ebdcb4d21aac979fb08315c3924685/image.png)

#FEATURES:
-Find where pieces belong on the board using OpenCV detection tool.
-Show 3 highest probability chances for puzzle piece location.
-Overlay transparent preview image over the board area.

#USAGE:
Piece Reference Size describes how big the puzzle piece is in Pixels. Each puzzle has a different piece size relative to the puzzle board size, the more pieces the puzzle has, the smaller the puzzle piece size. It is important that you calculate the piece size relative to the puzzle size using the instructions in README.docx Word document.

Overlay Duration slider tells the program how long to keep the green box overlay on the screen for. The Green Box overlay tells you where the program thinks the detected puzzle piece goes relative to the board area.

Full Board Transparency slider allows the user to change how transparent the preview image overlay is over the board area.

Paste and Search allows the user to screenshot a specific puzzle piece and the program will tell the user where the puzzle piece goes.

Show Full Overlay allows the user to show an overlay preview image of "puzzle.png" over the board area.

Set Board Area allows the user to set where the board is on their screen, it is important that you don't move your puzzle after setting the board area. It basically tells the program where the puzzle board frame is on your screen. So just highlight the frame of the puzzle exactly.

#IMPORTANT NOTE:
There are limitations with OpenCV detection, so accuracy is not always 100%.

P.S:
Solver7.py is the iteration with the new feature that tells you where the piece goes directly on the screen using a greenbox overlay. Solver4.py is the old iteration that worked pretty well and accurately but without a piece positioning overlay. You should play around with both and see which one you like the best. solver4.py has been tested pretty extensively, so try that one is all else fails.


