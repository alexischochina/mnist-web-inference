const path = require('path');
const CopyPlugin = require('copy-webpack-plugin');

module.exports = {
  mode: 'production',
  entry: './src/main.js',
  output: {
    filename: 'bundle.min.js',
    path: path.resolve(__dirname, 'dist'),
  },
  module: {
    rules: [
      {
        test: /\.js$/,
        exclude: /node_modules/,
        use: ['babel-loader'], // si tu veux transpiler
      },
      {
        test: /\.onnx$/,
        type: 'asset/resource',
        generator: {
          filename: 'models/[name][ext]'
        }
      }
    ],
  },
  plugins: [
    new CopyPlugin({
      patterns: [
        {
          from: path.resolve(__dirname, 'node_modules/onnxruntime-web/dist/*.wasm'),
          to: '[name][ext]'
        }
      ]
    })
  ],
  resolve: {
    conditionNames: ['onnxruntime-web-use-extern-wasm', 'import', 'module'],
  }
};
