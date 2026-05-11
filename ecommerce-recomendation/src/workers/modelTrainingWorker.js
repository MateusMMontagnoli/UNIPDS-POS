import 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js';
import { workerEvents } from '../events/constants.js';

console.log('Model training worker initialized');
let _globalCtx = {};
let _model = null;

const WEIGHTS = {
    category: 0.4,
    color: 0.3,
    price: 0.2,
    age: 0.1
}

// função para normalizar os dados entre 0 - 1
const normalize = (value, min, max) => (value - min) / ((max - min) || 1)


function makeContext(products, users) {
    const ages = users.map(u => u.age);
    const prices = products.map(c => c.price);

    const minAge = Math.min(...ages);
    const maxAge = Math.max(...ages);

    const minPrice = Math.min(...prices);
    const maxPrice = Math.max(...prices);

    const colors = [...new Set(products.map(c => c.color))];

    const categories = [...new Set(products.map(c => c.category))];

    const colorsIndex = Object.fromEntries
        (
            colors.map((color, index) => {
                return [color, index];
            })
        );

    const categoriesIndex = Object.fromEntries
        (
            categories.map((category, index) => {
                return [category, index];
            })
        );


    // calcular a media de idade dos compradores

    const avgAge = (minAge + maxAge) / 2;
    const ageSums = {}; // soma de todas as idades para cada produto
    const ageCounts = {}; // qtd total de produtos

    users.forEach(user => {
        user.purchases.forEach(purchase => {
            ageSums[purchase.name] = (ageSums[purchase.name] || 0) + user.age;
            ageCounts[purchase.name] = (ageCounts[purchase.name] || 0) + 1;
        })
    });


    const productAvgAgeNormalized = Object.fromEntries(
        products.map(product => {
            const avg = ageCounts[product.name] ?
                ageSums[product.name] / ageCounts[product.name] :
                avgAge;

            return [product.name, normalize(avg, minAge, maxAge)];
        })
    )

    return {
        products,
        users,
        productAvgAgeNormalized,
        colorsIndex,
        categoriesIndex,
        minAge,
        maxAge,
        minPrice,
        maxPrice,
        numCategories: categories.length,
        numColors: colors.length,
        // price, age, categorias e cores
        dimensions: 1 + 1 + categories.length + colors.length
    }
}

const oneHotWeighted = (index, length, weigth) =>
    tf.oneHot(index, length).cast('float32').mul(weigth);

function encodeProduct(product, context) {
    // normalizando dados e aplicação de peso
    const price = tf.tensor1d([
        normalize(product.price, context.minPrice, context.maxPrice) * WEIGHTS.price
    ]);

    const age = tf.tensor1d([
        (context.productAvgAgeNormalized[product.name] ?? 0.5) * WEIGHTS.age
    ]);

    const category = oneHotWeighted(
        context.categoriesIndex[product.category],
        context.numCategories,
        WEIGHTS.category
    );

     const color = oneHotWeighted(
        context.colorsIndex[product.color],
        context.numColors,
        WEIGHTS.color
    );

    return tf.concat1d(
        [price, age, category, color]
    )
}

function encodeUser(user, context) {
    if (user.purchases.length) {
        return tf.stack(
            user.purchases.map(product => encodeProduct(product, context))
        )
        .mean(0)
        .reshape([
            1,
            context.dimensions
        ])
    }

    return tf.concat1d(
        [
            tf.zeros([1]),
            tf.tensor1d([
                normalize(user.age, context.minAge, context.maxAge)
                * WEIGHTS.age
            ]),
            tf.zeros([context.numCategories]),
            tf.zeros([context.numColors])
        ]
    ).reshape([1, context.dimensions])


}

function createTrainingData(context) {
    const inputs = [];
    const labels = [];
    
    context
    .users
    .filter(user => user.purchases.length)
    .forEach(user => {
       const userVector = encodeUser(user, context).dataSync();

       context.products.forEach(product => {
            const productVector = encodeProduct(product, context).dataSync();

            const label = user.purchases.some(
                purchase => purchase.name === product.name ?
                1 : 
                0
            );

            inputs.push([...userVector, ...productVector]);
            labels.push(label);
        })

       
    });

    return {
        xs: tf.tensor2d(inputs),
        ys: tf.tensor2d(labels, [ labels.length, 1]),
        inputDimensions: context.dimensions * 2
        // tamanho = uservector + productVector
    }
}

async function configureNeuralNetAndTrain(trainData) {

    const model = tf.sequential();

    model.add(
        tf.layers.dense({
            inputShape: [trainData.inputDimensions],
            units: 128,
            activation: 'relu'
        })
    )

    model.add(
        tf.layers.dense({
            units: 64,
            activation: 'relu'
        })
    )

    model.add(
        tf.layers.dense({
            units: 32,
            activation: 'relu'
        })
    )

    // um nerorio apenas para retornar o score
    // sigmoid comprimir o retorno para 0 e 1
    model.add(
        tf.layers.dense({ units: 1, activation: 'sigmoid'})
    )

    model.compile({
        optimizer: tf.train.adam(0.01), 
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    })

    await model.fit(trainData.xs, trainData.ys, {
        epochs: 100,
        batchSize: 32,
        shuffle: true,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                postMessage({
                    type: workerEvents.trainingLog,
                    epoch: epoch,
                    loss: logs.loss,
                    accuracy: logs.acc
                });
            }
        }
    });

    return model;
}

async function trainModel({ users }) {
    console.log('Training model with users:', users)

    postMessage({ type: workerEvents.progressUpdate, progress: { progress: 50 } });

    const products = await (await fetch('/data/products.json')).json();

    const context = makeContext(products, users);

    context.productVectors = products.map(product => {
        return {
            name: product.name,
            meta: {...product},
            vector: encodeProduct(product, context).dataSync()
        }
    })

    
    const trainData = createTrainingData(context);

    _model = await configureNeuralNetAndTrain(trainData);



    _globalCtx = context;


    setTimeout(() => {
        postMessage({ type: workerEvents.progressUpdate, progress: { progress: 100 } });
        postMessage({ type: workerEvents.trainingComplete });
    }, 1000);


}
function recommend(user, ctx) {
    console.log('will recommend for user:', user)

    if (!_model) {
        return;
    }

    const context = _globalCtx;

    const userVector = encodeUser(user, context).dataSync();
    const inputs = context.productVectors.map(({vector}) => {
        return [ ...userVector, ...vector]
    })

    const inputTensor = tf.tensor2d(inputs)

    const predictions = _model.predict(inputTensor)

    const scores = predictions.dataSync();

    const recommendations = context.productVectors.map((entry, index) => {
        return {
            ...entry.meta,
            name: entry.name,
            score: scores[index]
        }
    })

    const sortedRecommendations = recommendations.sort((a,b) => b.score - a.score);

    postMessage({ 
        type: workerEvents.recommend,
        user,
        recommendations: sortedRecommendations
    });

    // postMessage({
    //     type: workerEvents.recommend,
    //     user,
    //     recommendations: []
    // });
}


const handlers = {
    [workerEvents.trainModel]: trainModel,
    [workerEvents.recommend]: d => recommend(d.user, _globalCtx),
};

self.onmessage = e => {
    const { action, ...data } = e.data;
    if (handlers[action]) handlers[action](data);
};
