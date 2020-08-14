# Notes:
* Add `Input` argument `channels_first` when using Z3 data and 
  change the input dimensions.
* You will have to manually add periodic edges for periodic padding before real run
* **requires** running the transform notebook to get the data into a format for the auto encoder

# Link Encodings

It is worth trying some different encodings. I can see some benefits 
and problems with the different approached and will probably just 
need to work through them by trying them. Here are some notes and
thoughts.

* Right now I believe the best thing to do is to physically represent links
  in a "location" We will have zeros "represent" the locations on the
  vertices and the centers of the plaquetts. 
* If you do the above you  still have some options for labeling the links.
  - You can use 1, 2, 3, 4, 5 to represent blank, up (N), down (S), right (E),
    left (W) respectively. In this case horizontal links can only  take values
    1, 4, 5. Vertical links can only take values 1, 2, 3
  - You could also use values 1, 2, 3 for blank, up/righ, down/left
    and the context of the position in the matrix which determines if it
    is a horizontal link or a vertical link would distinguish between
    up and right, and down and left.
* Of the above I think the second one would be nicer but I'm not sure if
  you could confuse the encoder to something important. Going to start with
  the first sub-bullet above and after I get something working try the second.
* Working with second approach as of 2020/08/14
* Transformed data is saved as a numpy file (pickle).