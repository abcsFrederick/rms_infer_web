import Vue from 'vue'
import App from './App.vue'
import AsyncComputed from 'vue-async-computed';
import GirderProvider from '@/plugins/girder';
import '@/plugins/resonantgeo';
import router from './router';
import 'material-design-icons-iconfont/dist/material-design-icons.css';
import VueYouTubeEmbed from "vue-youtube-embed";
Vue.use(VueYouTubeEmbed);
Vue.use(AsyncComputed);

Vue.config.productionTip = false

new Vue({
  provide: GirderProvider,
  router,
  render: h => h(App)
}).$mount('#app')


