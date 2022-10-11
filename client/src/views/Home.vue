<template>
<v-app>
  <div class="grey lighten-3 root">
    <v-container
      grid-list-xl
    >
      <v-layout row wrap>

 <v-dialog
        v-model="dialog"
        width="500"
        persistent
      >  
        <v-card>
          <v-card-title class="headline grey lighten-2">
            Research Use Policy
          </v-card-title>
  
          <v-card-text>
            <h>Please Sign In</h>
            This warning banner provides privacy and security notices consistent with applicable federal laws, directives, and other federal guidance for accessing this Government system, which includes (1) this computer network, (2) all computers connected to this network, and (3) all devices and storage media attached to this network or to a computer on this network.
This system is provided for Government-authorized use only.
Unauthorized or improper use of this system is prohibited and may result in disciplinary action and/or civil and criminal penalties.
Personal use of social media and networking sites on this system is limited as to not interfere with official work duties and is subject to monitoring.
By using this system, you understand and consent to the following:
The Government may monitor, record, and audit your system usage, including usage of personal devices and email systems for official duties or to conduct HHS business. Therefore, you have no reasonable expectation of privacy regarding any communication or data transiting or stored on this system. At any time, and for any lawful Government purpose, the government may monitor, intercept, and search and seize any communication or data transiting or stored on this system.
Any communication or data transiting or stored on this system may be disclosed or used for any lawful Government purpose.
            The algorithms and software provided on this site are
            intended for research purposes only.  This system has
            not been reviewed and approved by the Food and Drug
            Administration or any other US Federal agency for use
            in clinical applications.  The contained models are trained only on 
            Rhabdomyosarcoma tissue.  Testing images from any other disease
            is not supported and will not yield reliable results. 
          </v-card-text>
  
          <v-divider></v-divider>
  
          <v-card-actions>
            <v-spacer></v-spacer>
            <v-btn
              color="primary"
              text
              @click="autoLogin"
            >
              I accept
            </v-btn>
          </v-card-actions>
        </v-card>
      </v-dialog>


      <v-dialog
        v-model="loginGoogleDialog"
        width="600"
        persistent
      >  
        <v-card>
          <v-card-title class="headline grey lighten-2">
            Login using Google ID
          </v-card-title>
          <v-text-field
            label="Please enter your Google acccount ID"
            v-model="attemptedGoogleUserName"
          >
          </v-text-field>
          <v-text-field
             label="Please enter your Google account password"
             v-model="attemptedGoogleUserPassword"
             type='password'
          >
          </v-text-field>
          <v-divider></v-divider>
          <v-card-actions>
            <v-spacer></v-spacer>
            <v-btn
              color="primary"
              text
              @click="cilogonComponentLoginWrapper"
            >
              Login
            </v-btn>
          </v-card-actions>
        </v-card>
      </v-dialog>


      <v-dialog
        v-model="loginDialog"
        width="600"
        persistent
      >  
        <v-card>
          <v-card-title class="headline grey lighten-2">
            User Login
          </v-card-title>
          <v-text-field
            label="Please enter your user login"
            v-model="attemptedUserName"
          >
          </v-text-field>
          <v-text-field
             label="Please enter your password"
             v-model="attemptedUserPassword"
             type='password'
          >
          </v-text-field>
          <v-divider></v-divider>
          <v-card-actions>
            <v-spacer></v-spacer>
            <v-btn
              color="primary"
              text
              @click="girderComponentLoginWrapper"
            >
              Login
            </v-btn>
          </v-card-actions>
        </v-card>
      </v-dialog>


        <v-flex xs12 class="text-xs-center">
          <v-img class="mx-auto" width=600px :src="require('../assets/FNLCR-logo.png').default" />
          <v-img class="mx-auto" width=600px :src="require('../assets/nci-oncogenomics-logo.png').default" />
        </v-flex>
          <v-flex xs12>
            <v-card>
              <v-card-title class="headline grey lighten-2">
                Research Use Policy
              </v-card-title>
  
          <v-card-text>
            <ul>
              <li>
                This warning banner provides privacy and security notices consistent with applicable federal laws, directives, and other federal guidance for accessing this Government system, which includes (1) this computer network, (2) all computers connected to this network, and (3) all devices and storage media attached to this network or to a computer on this network.
              </li>
              <li>
                This system is provided for Government-authorized use only.
              </li>
              <li>
                Unauthorized or improper use of this system is prohibited and may result in disciplinary action and/or civil and criminal penalties.
              </li>
              <li>
                Personal use of social media and networking sites on this system is limited as to not interfere with official work duties and is subject to monitoring.
              </li>
              <li>
                By using this system, you understand and consent to the following:
                <ul>
                  <li>
                    The Government may monitor, record, and audit your system usage, including usage of personal devices and email systems for official duties or to conduct HHS business. Therefore, you have no reasonable expectation of privacy regarding any communication or data transiting or stored on this system. At any time, and for any lawful Government purpose, the government may monitor, intercept, and search and seize any communication or data transiting or stored on this system.
                  </li>
                  <li>
                    Any communication or data transiting or stored on this system may be disclosed or used for any lawful Government purpose.
                  </li>
                </ul>
              </li>
              <li>
                The algorithms and software provided on this site are
                intended for research purposes only.  This system has
                not been reviewed and approved by the Food and Drug
                Administration or any other US Federal agency for use
                in clinical applications.  The contained models are trained only on 
                Rhabdomyosarcoma tissue.  Testing images from any other disease
                is not supported and will not yield reliable results. 
              </li>
            </ul>
          </v-card-text>
        </v-card>
          <v-btn
              block
              :href="ITrustEndpoint"
            >
            {{ cilogonText }}
            </v-btn>
<!--
          <v-btn
              block
              @click="loginButton"
            >
            {{ loginText }}
            </v-btn>
-->
          <v-btn
              block
              @click="logout"
            >
            Log out
            </v-btn>
          </v-flex>

        <v-flex xs12 v-if="loggedIn">
          <span class="title">Applications</span>
        </v-flex>

        
        <v-flex v-if="loggedIn"
          v-for="(sample, i) in samples"
          :key="i"
          xs4
        >
          <v-card
            class="arborApp"
            flat
            tile
          >
            <v-img
              :src="sample.image"
              height="300px"
            />
            <div
              class="linkOverlay"
              v-on:click.stop="$router.push(sample.route)"
            >
              <div class="linkOverlayText body-2">
                {{ sample.label }}
                <div class="description" v-if="sample.description">
                  {{ sample.description }}
                </div>
              </div>
            </div>
          </v-card>
        </v-flex>

      </v-layout>
    </v-container>
  </div>
</v-app>
</template>

<script>

import { utils } from '@girder/components/src';
import optionsToParameters from '../optionsToParameters';
import { Authentication as GirderAuth } from "@girder/components/src/components";

components: {
    GirderAuth
}



export default {
  name: 'home',
  inject: ['girderRest'],
  data: () => ({
    smallScreen: false,
    dialog: false,
    loginDialog: false,
    girderLoginDialog: false,
    loginGoogleDialog: false,
    username: '',
    attemptedUserName: '',
    attemptedUserPassword: '',
    attemptedGoogleUserName: '',
    attemptedGoogleUserPassword: '',
    token: '',
    user: '',
    ADURL: '',
    cilogonText: 'Sign in',
    ITrustEndpoint: '',
    loginText: 'Please login here',
    samples: [
      {   
       label: 'Instructions',
       image: require('../assets/instructions.png').default,
       route: 'instructions',
       description: 'Quick start instructions on using this RMS application'
      },
      {   
       label: 'Whole Slide RMS Segmentation',
       image: require('../assets/RMS-WSI-segmentation.png').default,
       route: 'infer_wsi',
       description: 'Segment an entire H&E-stained Rhabdomyosarcoma WSI by uploading the slide for processing',
      },    
      {   
       label: 'MyoD1 Positive/Negative',
       image: require('../assets/myod1.png').default,
       route: 'myod1',
       description: 'Predict MYOD1 Positive/Negative from an H&E WSI',
      },   
      {   
       label: 'Survivability',
       image: require('../assets/survivability.png').default,
       route: 'survivability',
       description: 'Predict survivability from an H&E WSI',
      },   
    ],
  }),

// start with a 'research use only' dialog the first time 
// the page is rendered

mounted () {
  // 
  const resp = this.girderRest.get('/nciLogin/endpoint')
  if (resp.data) {
      this.ITrustEndpoint = resp.data
  }
},

created () {
  // ** test here they are not logged in.  If logged in, don't 
  // show the dialog again.
  this.username = localStorage.getItem('inferenceUser')
  if (this.username == null) {
      console.log('not logged in. showing dialog')
      this.dialog = true  
  } else {
   this.loginText = 'Logged in as user: '+this.username
  }
},

// this is used to control the rendering of the apps and anything
// else that isn't visible until the user has logged in
computed: {
  loggedIn() {
    // check login user or not
    const resp = this.girderRest.get('user/me')
    if (resp.data) {
      this.username = resp.data.login
      return true
    }
    return false
  },

 loggedIn_real() {
      if (this.username == null) {
        return false
      } else {
        return this.username.length>0
      }
    },
},


asyncComputed: {
  getURL() {

    // commented out since we aren't using CiLogin

    // let test = async (girderRest) => {
    //     // AD plugin API to fetch CI information/ returned URL(normally main app url) 
    //     // from your configuration(girder configure page.) after you set them.
    //     let inner = (await girderRest.get('/nciLogin/loginCallback', {})).data
    //     return inner;
    //   }
    //   test(this.girderRest).then(res => {
    //     // URL to CILogin page
    //     this.ADURL = res;
    //   })
  }
},

methods: {

autoLogin() {
  // close the announcement 
  this.dialog = false
  this.attemptedUserName = 'anonymous'
  this.attemptedUserPassword = 'letmein'
  // automatically log into girder so the apps are available
  this.loginFromGirderComponents( this.attemptedUserName,this.attemptedUserPassword)
  },

 // when the user clicks the button to login, open the login dialog

 loginButton() {
  console.log("open dialog to login the user");
  this.loginDialog = true
  //this.girderLoginDialog = true
 
  },

iTrustLogin() {
  console.log("open CILogon dialog to login the user");
  this.loginGoogleDialog = true
  },

 async girderComponentLoginWrapper() {
    const response =  this.loginFromGirderComponents( this.attemptedUserName,this.attemptedUserPassword)
 },


 async cilogonComponentLoginWrapper() {
    window.location = this.ADURL
 },


async loginFromGirderComponents(username, password, otp = null) {

    const GirderTokenLength = 64;
    const OauthTokenPrefix = '#girderToken=';
    const OauthTokenSuffix = '__';

    // Girder's custom headers
    const GirderToken = 'Girder-Token';
    const GirderOtp = 'Girder-OTP';
    const GirderAuthorization = 'Girder-Authorization';

    let auth;
    const headers = {
      [GirderToken]: null,
    };
 
    // try basic authentication  
    headers[GirderAuthorization] = `Basic ${window.btoa(`${username}:${password}`)}`;

    const resp = await this.girderRest.get('user/authentication', {
    headers, auth, withCredentials: false,
    })

    if (resp.status == 200){
      console.log('hooray, you are logged in')
      this.username = this.attemptedUserName
      this.loginDialog = false    
      this.token = resp.data.authToken.token;
      this.user = resp.data.user;
      this.username = this.attemptedUserName
      this.loginText = 'Logged in as user: '+this.username
      // set a local storage that will be persistent across the page reload 
      // this helps the app remember the user logged in when they come back to this
      // Home.vue view
      localStorage.setItem('inferenceUser',this.username)
    } else {
      console.log('sorry. try again')
      localStorage.removeItem('inferenceUser')
      this.username = ''
    }
    console.log('girder login response:',resp)
    return resp;
  },



  async logout() {
      if (!this.username) {
        return;
      } else {
        this.token = null;
        this.user = null;
        this.username = null;
        // remove from local storage so session is over in all pages
        localStorage.removeItem('inferenceUser')
        this.loginText = 'Please login here'
      }
    }

 }  // end methods

}

</script>

<style>
.root {
  display: flex;
  flex-flow: column;
  height: 100%;
  overflow-y: auto;
}

.buttonText {
  margin-right: 8px;
  text-transform: uppercase;
}

.arborApp {
  flex: 1;
  border-radius: 5px;
  overflow: hidden;
  cursor: pointer;
  box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);
}

.arborApp>div>div.v-image__image {
  background-size: contain;
}

.linkOverlay {
  position: absolute;
  top: 0;
  bottom: 0;
  width: 100%;
  z-index: 1;
  opacity: 0;
  background: rgba(255,255,255,0.2);
  font-size: 300%;
  text-align: center;
  font-weight: bolder;
  cursor: pointer;
  color: white;
}

.linkOverlay:hover {
  opacity: 1;
}

.description {
  font-size: 80%;
  color: lightgray;
}

.linkOverlayText {
  position: absolute;
  bottom: 0px;
  width: 100%;
  background: rgba(0,0,0,0.6);
  padding: 10px;
}
</style>
