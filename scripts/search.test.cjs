const assert = require('node:assert/strict');
const {test} = require('node:test');
const {search, termsFor, excerpt} = require('../search.js');
const articles = [
  {title: 'Neural Networks', fullTitle: 'A Guide to Neural Networks', url: '/neural-network', sections: [
    {title: 'Crop Yield', url: '/neural-network#crop-yield', text: 'Chosen weights transform fertilizer and rainfall into a continuous crop yield.'},
    {title: 'Activations', url: '/neural-network#activations', text: 'ReLU returns max(z, 0).'}]},
  {title: 'Rainfall Forecasts', url: '/forecasts', sections: [
    {title: 'Weather', url: '/forecasts#weather', text: 'Rainfall depends on the weather.'}]}
];
test('finds body-only concepts and links to their section', () => {
  const results = search(articles, 'fertilizer');
  assert.equal(results.length, 1);
  assert.equal(results[0].url, '/neural-network#crop-yield');
  assert.match(results[0].snippet, /fertilizer/);
});
test('requires every term and ranks title matches first', () => {
  assert.equal(search(articles, 'rainfall')[0].title, 'Rainfall Forecasts');
  assert.equal(search(articles, 'rainfall fertilizer').length, 1);
  assert.equal(search(articles, 'rainfall unknown').length, 0);
});
test('supports case, punctuation and Unicode without treating queries as regex', () => {
  assert.equal(search(articles, 'ReLU(z)')[0].url, '/neural-network#activations');
  assert.deepEqual(termsFor('学習 ReLU ReLU'), ['学習', 'relu']);
  assert.deepEqual(search(articles, '.*[]'), []);
  assert.deepEqual(search(articles, '   '), []);
});
test('returns only the best section per article and a passage around a late match', () => {
  assert.equal(search(articles, 'Neural Networks').length, 1);
  const text = 'A long introduction. '.repeat(20) + 'fertilizer raises the predicted yield. ' + 'More background. '.repeat(20);
  const passage = excerpt(text, ['fertilizer']);
  assert.match(passage, /fertilizer/);
  assert.ok(passage.startsWith('…'));
  assert.ok(passage.length <= 182);
});
