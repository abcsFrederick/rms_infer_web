<template>
  <v-app>
    <v-layout class="transform-view" row fill-height>
      <v-navigation-drawer permanent fixed style="width: 400px; min-width: 400px;">
        <v-toolbar dark flat color="primary">
          <v-toolbar-title class="white--text">RMS Survivability Classification</v-toolbar-title>
          </v-toolbar>
          <v-spacer/>
          <v-container fluid>
          <v-flex xs12>
              <v-btn
              outline
              block
                @click="loadSampleImageFile"
              >
              Use a Pre-Loaded Sample Image
              </v-btn>
            </v-flex>
          <v-flex xs12>
            <v-btn class="text-none" outline block @click='$refs.imageFile.click()'>{{ imageFileName || 'UPLOAD Whole Slide Image' }}</v-btn>
            <input
              type="file"
              style="display: none"
              ref="imageFile"
              @change="uploadImageFile($event.target.files[0])"
            >
          </v-flex>
      <!--
         <v-flex xs12>
            <v-btn class="text-none" outline block @click='$refs.segmentFile.click()'>{{ segmentFileName || '(optional) UPLOAD Segmentation Mask' }}</v-btn>
            <input
              type="file"
              style="display: none"
              ref="segmentFile"
              @change="uploadSegmentationFile($event.target.files[0])"
            >
          </v-flex>
      -->
      <v-switch
        v-model="fastmode"
        :label="`Enable Faster Approximate Result: ${fastmode.toString()}`"
      ></v-switch>

          <v-flex xs12>
            <v-btn
              block
              :class="{ primary: readyToRun }"
              :flat="readyToRun"
              :outline="!readyToRun"
              :disabled="!readyToRun"
              @click="run"
            >
              Calculate Risk Group
            </v-btn>
          </v-flex>

          <!-- <v-flex xs12>
            <v-btn
              block
              :class="{ primary: readyToDownload }"
              :flat="readyToDownload"
              :outline="!readyToDownload"
              :disabled="!readyToDownload"
              @click="downloadResults"
            >
              Download Results 
            </v-btn>
          </v-flex> -->

          <v-flex xs12>
            <v-btn
              block
              :class="{ primary: readyToDownload }"
              :flat="readyToDownload"
              :outline="!readyToDownload"
              :disabled="!readyToDownload"
              @click="reset"
            >
              Prepare For Another Image 
            </v-btn>
          </v-flex>
        </v-container>
      </v-navigation-drawer>
      <v-layout column justify-start fill-height style="margin-left: 400px">
          <v-card class="ma-4">
            <v-card-text>
              <b>
              For the survivability analsys, an ensemble model, constructed by combining a set of neural network models, has been trained to generate a risk prediction from an H&E whole slide image.
              This application runs the ensemble model on an uploaded image to generate risk categories derived from comoparing with our research cohort. 
              Uploaded images can be in Aperio (.svs) format or they can be pyramidal TIF files.
              <br><br>
              After selecting an image for upload, please be patient during the upload process. Once the input image is displayed below, please click the 
              "Calculate Risk Analysis" button to begin execution.  Execution may take longer than 30 minutes and up to several hours,
              depending on the size of the input image being provided and the computer used to execute the analysis.   
              <br><br>
              Since this analysis can take a very long time, an option is provided: Our full algorithm is an ensemble everage across 
              twenty separate models.   If instead you prefer to average only 25% of the models and receive an approximate answer back more
              quickly, just set the "Enable Faster Approximate Result" option to true.  Be aware this result won't be as accurate as the 
              full model.  However, it should return approximately four times faster than the full model.  
              <br><br>
              When the analysis is complete, the analysis result
              is shown as a downloadable chart below the images. Click the "three dot" menu at the top right of the chart 
              to download a copy of the chart.  If you would like to analyze additional images, 
              please just click "Prepare for Another Image" in between each segmentation operation. This tells the system 
              to reset and prepare to run again.  
              <br><br>
		          Thank you for trying our early release system for rhabdomyosarcoma whole slide analysis. 
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

        <div v-if="inputReadyForDisplay">
          <div  xs12 class="text-xs-center mb-4 ml-4 mr-4">
            <v-card class="mb-4 ml-4 mr-4">
              <v-card-text>Uploaded Image</v-card-text>
                <img :src="inputImageUrl" style="display: block; margin: auto"> 
              </v-card>
          </div>
        </div>

        <div v-if="segmentGenerationInProgress" xs12 class="text-xs-center mb-4 ml-4 mr-4">
          Segmenting the uploaded image.  Please wait for the output to show below.  This may take several minutes.
          <v-progress-linear :value="segmentProgress"></v-progress-linear>
        </div>

        <div v-if="segmentReadyForDisplay">
  	      <v-card class="mb-4 ml-4 mr-4">
            <v-card-text>Segmentation Mask</v-card-text>
               <img :src="segmentImageUrl" width="800" height="600" style="display: block; margin: auto"> 
          </v-card>
        </div>

        <div v-if="running" xs12 class="text-xs-center mb-4 ml-4 mr-4">
        Running Survivability Neural network inferencing.  Please wait for the output to show below.  This may take longer than 60 minutes.
          <v-progress-linear :value="surviveProgress"></v-progress-linear>
        </div>
        <div v-if="runCompleted" xs12 class="text-xs-center mb-4 ml-4 mr-4">
          Analysis Complete  ... 
        </div>
    
        <div v-show="runCompleted">
          <div  xs12 class="text-xs-center mb-4 ml-4 mr-4">
              Below is a chart comparing the survival predicted in the tested image when 
              compared to the images in our training cohort.  Click the elipsis icon at the top right 
              to save a copy of the chart to your local system.
          </div>
          <v-card  align="center" justify="center" class="mt-20 mb-4 ml-4 mr-4">     
            <div id="visM" ref="visModel" class="mt-20 mb-4 ml-4 mr-4"></div>
          </v-card>


          <v-card v-if="table.length > 0" class="mt-8 mb-4 ml-4 mr-4">
            <v-card-text>Probability (0 to 1) of MYOD1+ Mutation:</v-card-text>
            <json-data-table :data="table" />
          </v-card>
        </div>
        </v-layout>
    </v-layout>
  </v-app>
</template>

<script>

import { utils } from '@girder/components/src';
import { csvParse } from 'd3-dsv';
import scratchFolder from '../scratchFolder';
import pollUntilJobComplete from '../pollUntilJobComplete';
import optionsToParameters from '../optionsToParameters';
import JsonDataTable from '../components/JsonDataTable';
import vegaEmbed from 'vega-embed';
import UploadManager from '../utils/upload'

export default {
  name: 'survivability',
  inject: ['girderRest'],
  components: {
    JsonDataTable,
  },
  data: () => ({
    imageFile: {},
    imageFileName: '',
    segmentFile: {},
    segmentFileName: '',
    imagePointer: '',
    imageBlob: [],
    uploadedImageUrl: '',
    job: { status: 0 },
    readyToDisplayInput: false,
    readyToDisplaySegmentation: false,
    running: false,
    thumbnail: [],
    thumbnailInProgress: false,
    result: [],
    segmentResult: [],
    resultColumns: [],
    resultString:  '',
    runCompleted: false,
    uploadInProgress: false,
    segmentUploadInProgress: false,
    segmentGenerationInProgress: false,
    inputImageUrl: '',
    segmentImageUrl: '',
    outputImageUrl: '',
    table:[],
    inputDisplayed:  false,
    segmentDisplayed: false,
    outputDisplayed:  false,
    osd_viewer: [],
    cohortData: [],
    progressUpload: "0",
    segmentProgress: "0",
    surviveProgress: "0",
    stats: {},
    fastmode: true,
  }),
  asyncComputed: {
    scratchFolder() {
      return scratchFolder(this.girderRest);
    },
  },
  computed: {
    readyToRun() {
      return !!this.imageFileName && !this.running && this.inputDisplayed; 
    },
    readyToDownload() {
      return (this.runCompleted)
    },
    uploadIsHappening() {
      return (this.uploadInProgress)
    },
    segmentUploadIsHappening() {
      return (this.segmentUploadInProgress)
    },
    inputReadyForDisplay() {
      return this.inputDisplayed
    },
    segmentReadyForDisplay() {
      return this.segmentDisplayed
    }
  },

  methods: {

    // method here to create and display a thumbnail of an arbitrarily large whole slilde image.
    // This code is re-executed for each UI change, so the code is gated to only run once 

    async renderInputImage() {
       if (this.inputDisplayed == false) {

        // create a spot in Girder for the output of the REST call to be placed
          const outputItem = (await this.girderRest.post(
            `item?folderId=${this.scratchFolder._id}&name=thumbnail`,
          )).data

        // build the params to be passed into the REST call
        const params = optionsToParameters({
          imageId: this.imageFile._id,
          outputId: outputItem._id,
        });
        this.thumbnailInProgress = true
        // start the job by passing parameters to the REST call
        this.job = (await this.girderRest.post(
          `arbor_nova/wsi_thumbnail?${params}`,
        )).data;

          // wait for the job to finish
          await pollUntilJobComplete(this.girderRest, this.job, job => this.job = job);

          if (this.job.status === 3) {
            // pull the URL of the output from girder when processing is completed. This is used
            // as input to an image on the web interface
            this.thumbnail = (await this.girderRest.get(`item/${outputItem._id}/download`,{responseType:'blob'})).data;
            // set this variable to display the resulting output image on the webpage 
            this.thumbnailInProgress = false
            this.inputImageUrl = window.URL.createObjectURL(this.thumbnail);
          }

          console.log('render input finished')
	        this.inputDisplayed = true
	     }
    },


   async renderSegmentImage() {
       if (this.segmentDisplayed == false) {
          this.running = false;
          // pull the URL of the output from girder
          this.thumbnail = (await this.girderRest.get(`file/${this.segmentFile._id}/download`,{responseType:'blob'})).data;
          // set this variable to display the resulting output image on the webpage 
          this.segmentImageUrl = window.URL.createObjectURL(this.thumbnail);
          console.log('render segment finished')
          this.segmentDisplayed = true
          this.segmentUploadInProgress = false
	     }
    },


    extractStatsFromJobLog(job) {
        this.job = job
        // pick out the last print message from the job

        var last_element = job.log[job.log.length - 1];
          if (last_element) {
            console.log('last_element:',last_element)
            let stats = JSON.parse(last_element)
            console.log('stats as json:',stats)
            console.log('secondBest:',stats.secondBest)
            return stats
        }
        return {}
      },

    updateJobStatus(job) {
        this.job = job
        // pick out the last print message from the job

        var last_element = job.log[job.log.length - 1];
        if (last_element) {
            var lastIndex = last_element.lastIndexOf('\n')
            let progressSnippet = last_element.substring(0,lastIndex)
            // for some reason in this app, we receive a multiline string, so split it 
            // and explore each component
            //console.log('snippet:',progressSnippet)
            var splitLog = progressSnippet.split('\n')
            console.log('split:',splitLog)
            // look through each component looking for the progress update
            for (let i = 0; i < splitLog.length; i++) {
              var testString = splitLog[i].substring(0,8)
              if (testString == 'progress') {
                  // we found the progress printout, get the value and update progress
                  var progValue = splitLog[i].substring(10,splitLog[i].length)
                  console.log('found progress:',progValue)
                  this.surviveProgress = progValue
              }
            }
        }
      },

    async run() {
      this.errorLog = null;

      // there may be a case where the user uploaded only a target image and not a corresponding segmentation.
      // if this is the case, then fire off the segmentation algorithm on demand and then load the resulting 
      // segmentation into the web page to prepare for the survivability model execution. 

      if (this.segmentFileName.length == 0) {
        // we don't have a segmentation file yet, we need to generate one
        console.log('generating segmentation.  please wait.')
        await this.generateSegmentation()
      }

      // create a spot in Girder for the output of the survivability REST call to be placed
      console.log('do I need this output item???')
      const outputItem = (await this.girderRest.post(
        `item?folderId=${this.scratchFolder._id}&name=result`,
      )).data

      // build the params to be passed into the REST call. This is a var instead of const, because it is reused.  We 
      // aren't returning a girder item, we only return a dictionary of statistics, so there is no result hook here. 


      var params = optionsToParameters({
        imageFileName: this.imageFileName,
        imageId: this.imageFile._id,
        segmentFileName: this.segmentFileName,
        segmentId: this.segmentFile._id,
        fastmode: this.fastmode,
        statsId: outputItem._id
      });
      // start the job by passing parameters to the REST call
      this.running = true;
      console.log('starting backend inferencing with params',params)
      //this.job = (await this.girderRest.post(
      //  `survivability?${params}`,
      //)).data;

      this.job = (await this.girderRest.post(
        `arbor_nova/survivability?${params}`,
      )).data;

      // wait for the job to finish and then download the cohort table
      await pollUntilJobComplete(this.girderRest, this.job, this.updateJobStatus);
      console.log(this.job)

      // 3 is the status returned by the plugin as job.SUCCESS
      if (this.job.status === 3) {
        // the survivability model ran successfully, so capture the hazard prediction value that came back
        // from the model and parse so we can draw a rendering chart on the web page showing how the image
        // compared with our previous cohort

        this.running = false;
        this.runCompleted = true;
        this.result = (await this.girderRest.get(`item/${outputItem._id}/download`,{responseType:'text'})).data;
        this.stats = this.result
        //this.stats = this.extractStatsFromJobLog(this.job)
        console.log('stats:',this.stats)
        
        // now fetch the cohort that we need to compare against from girder storage.  This way the cohort
        // can be updated by changing the girder contents instead of hard-coding the web app.

        // create a spot in Girder for the output of the REST call to be placed
        const cohortItem = (await this.girderRest.post(
          `item?folderId=${this.scratchFolder._id}&name=cohort`,
        )).data

        // build the params to be passed into the REST call
        var params = optionsToParameters({
          cohortName: 'survivability',
          outnameId: cohortItem._id,
        });
        console.log('params:',params)
        // start the job by passing parameters to the REST call.  This job will return 
        // a table of cohort values for the 'survivability' cohort.
        this.job = (await this.girderRest.post(
          `arbor_nova/cohort?${params}`,
        )).data;

        // wait for the job to finish and then download the cohort table
        await pollUntilJobComplete(this.girderRest, this.job, job => this.job = job);
        this.cohortData = csvParse((await this.girderRest.get(`item/${cohortItem._id}/download`)).data);

        // convert string values to float values for vega spec in place in variable this.cohortData
        for (let index = 0; index < this.cohortData.length; index++) {
          var element = this.cohortData[index];
          element['Hazard Prediction'] = parseFloat(element['Hazard Prediction'])
          this.cohortData[index] = element
          
        }
        console.log('returned cohort',this.cohortData)

        // write the prediction to the web console 
        console.log('prediction', this.stats, this.stats.secondBest)

        // the data needs to be stored in a local variable to render using vegaEmbed
        let secondBest = this.stats.secondBest
        let myCohortData = this.cohortData

        // add this image result as an "Uploaded" type
        //myCohortData.push({'Patient ID':'Uploaded','Risk Group': 'Uploaded','Event Free Survival': "50",'Hazard Prediction': secondBest.toString()})
        //console.log('amended cohort',myCohortData)
        //console.log('amended cohort as string:',JSON.stringify(myCohortData))

        // build the spec here.  Inside the method means that the data item will be available.  This spec is a boxplot of the cohort
        // of data with a vertical line superimposed with the value of the analysis for this particular image. 
        var vegaLiteSpec = {
            "title": "Predicted Survivability of the Uploaded Image Compared to Our Cohort",
            "height":250,
            "width": 500,
            "data": {
              "values": myCohortData },
            "layer": [
            {
              "mark": "boxplot",
              "encoding": {
                "x": {
                  "field": "Hazard Prediction",
                  "type": "quantitative",
                  "scale": {"zero": true}
                },
                "y": {
                  "field": "Risk Group",
                  "type": "nominal"
                },
                "size": {"value":40},
                "color": {
                  "field": "Risk Group",
                  "type":"nominal",
                  "scale": {"domain":["High","Intermediate","Low","Uploaded"],"range": ["red","green","blue","orange"]}
                }
                }
              }, 
              {
                  "mark": "rule",
                  "data": {
                    "values": [
                      {"Risk Group": "Uploaded", "Hazard Prediction": secondBest,  "Event Free Survival":"50"}
                    ]
                  },
                  "encoding": {
                    "x": {
                      "field": "Hazard Prediction",
                      "type": "quantitative"
                    },
                    "color": {"value": "orange"},
                    "size": {"value": 4}
                  }
                },
                {
                  "mark": {
                      "type":"text",
                      "fontSize": 14,
                      "dx": -25
                      },
                  "data": {
                    "values": [
                      {"Risk Group": "Uploaded", "Hazard Prediction": secondBest, "Event Free Survival":"50"}
                    ]
                  },
                  "encoding": {
                    "text": {"field":"Hazard Prediction","type":"quantitative"},
                    "x": {
                      "field": "Hazard Prediction",
                      "type": "quantitative"
                    },
                    "y": {
                        "field": "Risk Group",
                        "type": "ordinal"
                        },
                    "color": {"value": "orange"}
                  }
                }
            ]
          };
        // render the chart with options to save as PNG or SVG, but other options turned off
        vegaEmbed(this.$refs.visModel,vegaLiteSpec,
                  {padding: 10, actions: {export: true, source: false, editor: false, compiled: false}});
    }

  },

    async uploadImageFile_original(file) {
      if (file) {
        this.runCompleted = false;
        this.imageFileName = file.name;
        const uploader = new utils.Upload(file, {$rest: this.girderRest, parent: this.scratchFolder});
        this.uploadInProgress = true;
        this.imageFile = await uploader.start();
        // display the uploaded image on the webpage
        this.uploadInProgress = false;
	      console.log('displaying input image...');
        //this.imageBlob = (await this.girderRest.get(`file/${this.imageFile._id}/download`,{responseType:'blob'})).data;
        //this.uploadedImageUrl = window.URL.createObjectURL(this.imageBlob);
	      //console.log('createObjURL returned: ',this.uploadedImageUrl);
        this.readyToDisplayInput = true;
        this.renderInputImage();
      }
    },


    // display the progress of the image upload operation; called by uploadImageFile method during the upload process
    async receiveUploadProgress(status) {
      //console.log(status.current,status.size)
      this.progressUpload = (status.current/status.size*100.0).toString()
      console.log('upload progress:',this.progressUpload)
    },

    // upload a file in multiple chunks to support large WSI files; a callback is supported to show progress
    async uploadImageFile(file) {
      if (file) {
        this.runCompleted = false;
        this.imageFileName = file.name;
        this.uploadInProgress = true;
        // this upload manager splits the file into smaller transfers to allow uploading a large file with a progress bar
        const uploader = new UploadManager(file, {$rest: this.girderRest, parent: this.scratchFolder,progress: this.receiveUploadProgress});
        this.imageFile = await uploader.start();
        // display the uploaded image on the webpage
	      console.log('displaying input image...');
        this.readyToDisplayInput = true;
        this.uploadInProgress = false;
        this.renderInputImage();
      }
    },



async uploadSegmentationFile(file) {
      if (file) {
        this.runCompleted = false;
        this.segmentFileName = file.name;
        const uploader = new utils.Upload(file, {$rest: this.girderRest, parent: this.scratchFolder});
        this.segmentUploadInProgress = true;
        this.segmentFile = await uploader.start();
        // display the uploaded image on the webpage
        this.segmentUploadInProgress = false;
	      console.log('received segmentation image...');
        //this.imageBlob = (await this.girderRest.get(`file/${this.imageFile._id}/download`,{responseType:'blob'})).data;
        //this.uploadedImageUrl = window.URL.createObjectURL(this.imageBlob);
	      //console.log('createObjURL returned: ',this.uploadedImageUrl);
        this.readyToDisplaySegmentation = true;
        this.renderSegmentImage();
      }
    },

    // this routine is called when the user indicates they want to run the analysis, but there is no
    // segmentation file specifically loaded.  In this case, run the segmentation model and upload the result
    // to the UI



  updateSegmentJobStatus(job) {
      this.job = job
      // pick out the last print message from the job

      var last_element = job.log[job.log.length - 1];
        if (last_element) {
        //console.log(last_element)
        let lastIndex = last_element.lastIndexOf('\n')
        //console.log('lastindex:',lastIndex)
        let progressSnippet = last_element.substring(lastIndex)
        //console.log(progressSnippet)
        //console.log(progressSnippet.substring(1,9))
        //console.log(progressSnippet.substring(1,2))
        // if this is a progress update string, then extract the percentage done and update the state variable
        if (progressSnippet.substring(1,9)=='progress') {
          // starting at this position, is the string of the value to update the progress bar
          this.segmentProgress = progressSnippet.substring(11)
          //console.log('segment percentage:',this.segmentProgress)
        }
      }
    },



    async generateSegmentation() {
        this.runCompleted = false;
        this.segmentGenerationInProgress = true;
        this.segmentFileName = 'Generating Segmentation...'

        // create a spot in Girder for the output of the REST call to be placed
        const outputItem = (await this.girderRest.post(
          `item?folderId=${this.scratchFolder._id}&name=segmentResult`,
        )).data

        // create a spot in Girder for the output of the REST call to be placed
        const statsItem = (await this.girderRest.post(
          `item?folderId=${this.scratchFolder._id}&name=stats`,
        )).data

        // build the params to be passed into the REST call
        const params = optionsToParameters({
          imageId: this.imageFile._id,
          outputId: outputItem._id,
          statsId: statsItem._id
        });
        // start the job by passing parameters to the REST call
        this.job = (await this.girderRest.post(
          `arbor_nova/infer_rms_map?${params}`,
        )).data;

        // wait for the job to finish
        await pollUntilJobComplete(this.girderRest, this.job, this.updateSegmentJobStatus);


        // display the uploaded image on the webpage
        this.segmentGenerationInProgress = false;
	      console.log('calculated segmentation image...');
        this.segmentResult = (await this.girderRest.get(`item/${outputItem._id}/download`,{responseType:'blob'})).data;
        this.segmentFileName = 'Generated Segmentation'

         // set this variable to display the resulting output image on the webpage 
         
        this.readyToDisplaySegmentation = true;
        this.segmentImageUrl = window.URL.createObjectURL(this.segmentResult);

        console.log('generate segment finished')
        this.segmentDisplayed = true

        // ACTION - get the item's file 
        //  lookup right here 
        this.segmentFile = (await this.girderRest.get(`item/${outputItem._id}/files`)).data[0];
        this.renderSegmentImage()

        return this.segmentImageUrl
    },


    // loading a sample image means loading the WSI and a corresponding segmentation.  Both of these are done
    // here.  This requires girder to be pre-loaded with image names that match the patterns here. 

    async loadSampleImageFile() {
          console.log('load sample image')
          this.runCompleted = false;
          this.uploadInProgress = true;
          this.imageFileName = 'Sample_WSI_Image.svs'
          const params = optionsToParameters({
                q: this.imageFileName,
                types: JSON.stringify(["file"])
              });
          // find the sample image already uploaded in Girder
          this.fileId = (await this.girderRest.get(
            `resource/search?${params}`,
          )).data["file"][0];

          console.log('displaying sample input stored at girder ID:',this.fileId);
          this.imageFile = this.fileId
          this.inputDisplayed == false;
          this.uploadInProgress = false;
          this.renderInputImage();


          // now get the segmentation image to match the WSI
          this.segmentUploadInProgress = true
          this.segmentFileName = 'Sample_WSI_Segmentation.png'
          const params2 = optionsToParameters({
                q: this.segmentFileName,
                types: JSON.stringify(["file"])
              });
          // find the sample image already uploaded in Girder
          this.fileId = (await this.girderRest.get(
            `resource/search?${params2}`,
          )).data["file"][0];

          console.log('displaying sample segmentation stored at girder ID:',this.fileId);
          this.segmentFile = this.fileId
          this.segmentDisplayed = false;
          this.readyToDisplayInput = true;
          this.renderSegmentImage();
          },


    // download the segmentation image result when requested by the user
    async downloadResults() {
        const url = window.URL.createObjectURL(this.result);
	      console.log("url:",url)
        const link = document.createElement('a');
        link.href = url;
        link.setAttribute('download', 'infer_results.png') 
        document.body.appendChild(link);
        link.click();
	      document.body.removeChild(link);
    },

    // reload the page to allow the user to process another image.
    // this clears all state and image displays. The scroll command
    // resets the browser to the top of the page. 
    reset() {
      window.location.reload(true);
      window.scrollTo(0,0);
    },
  }
}
</script>
