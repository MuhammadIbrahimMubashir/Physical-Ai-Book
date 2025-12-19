// @ts-check
const config = {
  title: 'Physical AI & Humanoid Robotics',
  tagline: 'Learn Robotics and Physical AI step by step',
  url: 'https://MuhammadIbrahimMubashir.github.io',
  baseUrl: '/Physical-Ai-Book/',
  
  // <<< CHANGE THESE LINES >>>
  onBrokenLinks: 'warn',          // changed from 'throw' to 'warn'
  onBrokenMarkdownLinks: 'warn',  // keep as 'warn'

  favicon: 'img/favicon.ico',
  organizationName: 'MuhammadIbrahimMubashir', 
  projectName: 'Physical-Ai-Book',
  presets: [
    [
      '@docusaurus/preset-classic',
      {
        docs: {
          path: 'docs',
          routeBasePath: '/',
          sidebarPath: require.resolve('./sidebars.js')
        },
      }
    ]
  ]
};

module.exports = config;
