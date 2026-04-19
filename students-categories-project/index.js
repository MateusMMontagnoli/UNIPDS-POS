import tf from '@tensorflow/tfjs-node';


async function trainModel(inputXs, outputYs) {
    const model = tf.sequential();

    // primeira camada da rede
    // entrada de 7 posições
    // 1 idade
    // 3 cores
    // 3 localizações

    // 80 neuronios
    // ReLU filtra apenas os dados interessantes seguire pela rede
    // Se o calculo do neuronio for 0 ou negativo descatar
    model.add(tf.layers.dense({ inputShape: [7], units: 80, activation: 'relu'}));

    // ultima camada
    // 3 neuoronios
    // um para cada categoria (premium, medium, basic)
    // activation é softmax
    model.add(tf.layers.dense({units: 3, activation: 'softmax'}));

    // compilando modelo
    // loss: categoricalCrossentropy
    // compara o que o modelo acha com a resposta certa
    // optimizer: Adam (Adaptive Moment Estimation)
    // aprender com o historico de erro e acertos para melhoras os pesos
    // quanto mais distante da previsao do modelo da resposta correta
    // maior o erro (loss)
    // Exemplo classico: classificação de imagem, recomendação, categorização de usuário
    model.compile({ optimizer: 'adam' , loss: 'categoricalCrossentropy', metrics: ['accuracy']});

    // treinamento
    // verbose: desabilita os logs internos e apenas utiliza os callbacks
    // epochs: quantidade de veses que vai rodar o dataset
    // shuffle: embaralhamento de dados para evitar viés
    await model.fit(
        inputXs,
        outputYs,
        {
            verbose: 0,
            epochs: 100,
            shuffle: true,
            callbacks: {
                onEpochEnd: (epoch, log) => console.log(`Epoch: ${epoch}: loss = ${log.loss}`)
            }
        }
    )

    return model
}

// Exemplo de pessoas para treino (cada pessoa com idade, cor e localização)
// const pessoas = [
//     { nome: "Erick", idade: 30, cor: "azul", localizacao: "São Paulo" },
//     { nome: "Ana", idade: 25, cor: "vermelho", localizacao: "Rio" },
//     { nome: "Carlos", idade: 40, cor: "verde", localizacao: "Curitiba" }
// ];

// Vetores de entrada com valores já normalizados e one-hot encoded
// Ordem: [idade_normalizada, azul, vermelho, verde, São Paulo, Rio, Curitiba]
// const tensorPessoas = [
//     [0.33, 1, 0, 0, 1, 0, 0], // Erick
//     [0, 0, 1, 0, 0, 1, 0],    // Ana
//     [1, 0, 0, 1, 0, 0, 1]     // Carlos
// ]

// Usamos apenas os dados numéricos, como a rede neural só entende números.
// tensorPessoasNormalizado corresponde ao dataset de entrada do modelo.
const tensorPessoasNormalizado = [
    [0.33, 1, 0, 0, 1, 0, 0], // Erick
    [0, 0, 1, 0, 0, 1, 0],    // Ana
    [1, 0, 0, 1, 0, 0, 1]     // Carlos
]

// Labels das categorias a serem previstas (one-hot encoded)
// [premium, medium, basic]
const labelsNomes = ["premium", "medium", "basic"]; // Ordem dos labels
const tensorLabels = [
    [1, 0, 0], // premium - Erick
    [0, 1, 0], // medium - Ana
    [0, 0, 1]  // basic - Carlos
];

// Criamos tensores de entrada (xs) e saída (ys) para treinar o modelo
const inputXs = tf.tensor2d(tensorPessoasNormalizado)
const outputYs = tf.tensor2d(tensorLabels)

const models = trainModel(inputXs, outputYs)