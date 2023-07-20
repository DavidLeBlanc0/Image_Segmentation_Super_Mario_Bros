# Image_Segmentation_Super_Mario_Bros

Experiment tools for my master's research on applying image segmentation to deep reinforcement learning in Super Mario Bros. Below are some of the figures that I generated with this code. If you are interested in reading more, you can find a complete explanation in my thesis: [Applying Image Segmentation To Deep Reinforcement Learning In Video Games](https://scholar.acadiau.ca/islandora/object/theses%3A3872?solr_nav%5Bid%5D=b471c82c429104241851&solr_nav%5Bpage%5D=0&solr_nav%5Boffset%5D=8). The manual tool I created for labelling data is also available on my [GitHub page](https://github.com/DavidLeBlanc0/Image_Segmentation_Grid_Labeller).


![man_auto_vid](https://github.com/DavidLeBlanc0/Image_Segmentation_Super_Mario_Bros/assets/57907981/dc6d0e99-db8d-484e-904f-506179e8698b)
*The predicted segmentation for the segmenting models on an image from a test video. From left to right: the image
being used as input, the manual model’s prediction, the small automatic model’s prediction, and the large automatic model’s
prediction. The bottom source image depicts Mario between two pipes (ground) and a Piranha Plant (hazard).*

![automatic_confusion](https://github.com/DavidLeBlanc0/Image_Segmentation_Super_Mario_Bros/assets/57907981/5ac00acf-e069-49be-8337-6b331993dad6)
*The confusion matrix of a CNN trained to convert frames of Super Mario Bros. to a segmented grid.*

![autoencoder_preds](https://github.com/DavidLeBlanc0/Image_Segmentation_Super_Mario_Bros/assets/57907981/406c7cc6-c206-4d49-a2fa-f091f4a01137)

*The results of an autoencoder reconstructing a frame from Super Mario Bros*
