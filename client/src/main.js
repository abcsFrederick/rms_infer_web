import Vue from 'vue'
import App from './App.vue'
import GirderProvider from '@/plugins/girder';
import '@/plugins/resonantgeo';
import router from './router';
import 'material-design-icons-iconfont/dist/material-design-icons.css';
import VueYouTubeEmbed from "vue-youtube-embed";
Vue.use(VueYouTubeEmbed);

Vue.config.productionTip = false

new Vue({
  provide: GirderProvider,
  router,
  render: h => h(App)
}).$mount('#app')


