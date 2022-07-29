<template>
  <v-app>
    <v-layout class="transform-view" row fill-height>
      <v-navigation-drawer permanent fixed style="width: 400px; min-width: 400px;">
        <v-toolbar dark flat color="primary">
          <v-toolbar-title class="white--text">Instructions for this Tool</v-toolbar-title>
        </v-toolbar>
        <v-spacer/>
        <v-container fluid>
        </v-container>
      </v-navigation-drawer>
      <v-layout column justify-start fill-height style="margin-left: 400px">
          <v-card class="ma-4">
            <v-card-text>
              <b>This interactive web tool provides access to several pre-trained Rhabdomyosarcoma models 
                through a web interface.  This application can be installed natively on a physical
                or virtual machine.  The application is also available as a downloadable docker
                container.  
              <br><br>
    This web interface is organized as several mini-applications: (instructions, segmentation, MYOD1 mutation, and survivability),
     each of which has its own panel and instructions.  Below you will find demonstration videos 
     about each of the RMS mini-applications included in this system.  To get back to the main interface anytime, 
     just click the browser's back button. 
              <br><br>
		We are delighted that you are trying our early release system for rhabdomyosarcoma analysis. Thank you.  
		</b>
            </v-card-text>
            
          </v-card>
           <div v-if="uploadIsHappening" xs12 class="text-xs-center mb-4 ml-4 mr-4">
           Image Upload in process...
           <v-progress-linear :value="progressUpload"></v-progress-linear>
          </div>

          <div v-if="thumbnailInProgress" xs12 class="text-xs-center mb-4 ml-4 mr-4">
            Generating a thumbnail of the uploaded image
            <v-progress-linear indeterminate=True></v-progress-linear>
          </div>

        <div  xs12 class="text-xs-center mb-4 ml-4 mr-4">
  	       <v-card class="mb-4 ml-4 mr-4">
            <v-card-text>
              Below is a demonstration video of the Rhabdomyosarcoma segmentation model
              <br></br>
              Our segmentation model, which identifies regions containing ARMS, ERMS, necrosis, 
              and stroma in whole slide images, utilizes a standard UNet model as implemented 
              by the Segmentation Models package for pytorch.   The UNet employs transfer learning, 
              initializied with weights from prior training using the ImageNet database,  
              before being trained using images from our cohort.  

            </v-card-text>
               <div>
                <section>
                  <youtube
                    :video-id="SegmentvideoId"
                    :player-width="500"
                    :player-height="300"
                    @ready="ready"
                    @playing="playing"
                  ></youtube>
                </section>
              </div>
            </v-card>
        </div> 
   
        <div  xs12 class="text-xs-center mb-4 ml-4 mr-4">
  	       <v-card class="mb-4 ml-4 mr-4">
            <v-card-text>
              Below is a demonstration video of the RMS MYOD1 mutation prediction  model
             <br></br>
              The MYOD1 classification network uses pre-trained ResNet50 models.  Similar to the segmentation 
              model, the ImageNet dataset was used to provide the pre-trained weights for the model 
              prior to additional training on our cohort of images to predict MYOD1 mutation.  
              The method used in this application is to sample four thousand small patches containing 
              RMS lesions from random locations within the image being analyzed.  Our segmentation 
              model (described above) is used to detect the presence of a form of RMS in the 
              patches. These patches are then input to the ResNet50 model to generate a prediction 
              score.  To improve accuracy, our final prediction is taken as the average output value 
              from an ensemble of three separately trained ResNet50 networks.  
            </v-card-text>
              <div>
                <section>
                  <youtube
                    :video-id="MYOD1videoId"
                    :player-width="500"
                    :player-height="300"
                    @ready="ready"
                    @playing="playing"
                  ></youtube>
                </section>
              </div>
            </v-card>
        </div> 

        <div  xs12 class="text-xs-center mb-4 ml-4 mr-4">
  	       <v-card class="mb-4 ml-4 mr-4">
            <v-card-text>
              Below is a demonstration video of the RMS Survivability prediction model
              <br></br>
              The Survivability model uses the same approach as the MYOD1 mutation model,
               by extracting four thousand patches from the source image and generating 
               a prediction score from the neural network output based on these input 
               patches.  The classifier model consists of an ensemble of ResNet18 deep 
               learning networks, each pre-trained on ImageNet.  To improve accuracy, 
               our final prediction is calculated as the average output value from an 
               ensemble of twenty separately trained ResNet18 networks. 
            </v-card-text>
              <div>
                <section>
                  <youtube
                    :video-id="SurvideoId"
                    :player-width="500"
                    :player-height="300"
                    @ready="ready"
                    @playing="playing"
                  ></youtube>
                </section>
              </div>
            </v-card>
        </div> 


      </v-layout>
    </v-layout>
  </v-app>
</template>

<script>
import { getIdFromURL } from "vue-youtube-embed";
let SegmentvideoId = getIdFromURL("https://www.youtube.com/watch?v=HImUo94BLn8");
let MYOD1videoId = getIdFromURL("https://www.youtube.com/watch?v=AF5xpUWVX8c");
let SurvideoId = getIdFromURL("https://www.youtube.com/watch?v=zvL1UzGj6Qw");

// vue-youtube-embed example found here:
// https://www.nightprogrammer.com/vue-js/how-to-embed-a-youtube-video-and-use-controls-in-vue-js-example/

export default {
  name: 'instructions',
  components: {
  },
  data: () => ({
    SegmentvideoId,
    MYOD1videoId,
    SurvideoId

  }),
  asyncComputed: {

  },
  computed: {
    
  },

  methods: {
    ready(event) {
      this.player = event.target;
    },
    playing(event) {
      console.log("playing");
    },
    change() {
      //this.videoId = "use another video id";
    },
    stop() {
      this.player.stopVideo();
    },
    pause() {
      this.player.pauseVideo();
      console.log("paused");
    },
    play() {
      this.player.playVideo();
      console.log("paused");
    },

  }
}
</script>


<style>
#app {
  font-family: "Avenir", Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  margin: 15px 10px;
}
.np-ib {
  display: inline-block;
}
.np-button {
  margin-top: 10px;
  background: #0051ff;
  color: #ffffff;
  width: 80px;
  margin-right: 10px;
  text-align: center;
  padding: 4px 8px;
  border-radius: 4px;
  cursor: pointer;
  transition: all 0.3s;
}
.np-button:hover {
  background: #4e86ff;
  transition: all 0.3s;
}
.np-credits {
  font-size: 12px;
  padding-bottom: 14px;
}
</style>