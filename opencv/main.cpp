#include <fstream>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

// Tipo de coleta [1: Webcam, 2: Arquivo de vídeo]
std::string tipo;
// Path da configuração da rede neural
std::string configCaminho =
    "/home/piemontez/Projects/piemontez/datasets/yolov3.cfg";
// Path da rede neural treinada
std::string modelCaminho =
    "/home/piemontez/Projects/piemontez/datasets/yolov3.weights";
// Path dos rotulos treinados
std::string labelCaminho;

cv::dnn::Backend backendPreferencial = cv::dnn::DNN_BACKEND_CUDA;
cv::dnn::Target targetPreferencial = cv::dnn::DNN_TARGET_CUDA;

float confThreshold = 0.6;
float nmsThreshold = 0;
cv::Scalar mean = 0;
bool swapRB = true;
int inpWidth = 640;
int inpHeight = 480;
float scale = 0.00392;

std::vector<std::string> rotulos;

void exibirConfiguracoesAcessiveis();
bool coletarParametros();
void carregarConfiguracoes();
void alterarLimiarConfianca(int pos, void *);

inline void preprocess(const cv::Mat &frame, cv::dnn::Net &net,
                       cv::Size inpSize, float scale, const cv::Scalar &mean,
                       bool swapRB);
void postprocess(cv::Mat &frame, const std::vector<cv::Mat> &out,
                 cv::dnn::Net &net, int backend);
void drawPred(int classId, float conf, int left, int top, int right, int bottom,
              cv::Mat &frame);

int main() {
  // Exibe tipos de processamentos suportados pelo seu computador
  exibirConfiguracoesAcessiveis();

  // Carrega os parametros informados ou questiona o usuário
  if (!coletarParametros()) {
    return 1;
  }

  // Carregas as configurações conforme parametros informados
  // Carrega os rótulos da rede neural
  carregarConfiguracoes();

  // Carrega a rede neural com a função readNet
  // O método readNet identifica o modelo de rede neura informado e redireciona
  // para a função adequada readNetFromCaffe, readNetFromTensorflow,
  // readNetFromTorch, readNetFromDarknet.
  cv::dnn::Net net = cv::dnn::readNet(modelCaminho, configCaminho);
  // Tipo de processamento interno preferencial: Opencv, VKCOM, Halide, ...
  net.setPreferableBackend(backendPreferencial);
  // Tecnica de computacao dos dados: CPU, CUDA,....
  net.setPreferableTarget(targetPreferencial);
  std::vector<cv::String> outNames = net.getUnconnectedOutLayersNames();

  static const std::string nomeJanela =
      "Deep learning object detection in OpenCV";
  {
    // Cria janela para exibição do vídeo sendo processado
    cv::namedWindow(nomeJanela, cv::WINDOW_NORMAL);

    // Adiciona controlador slider para alterar
    // limiar de confiaça da rede neural
    int initialConf = (int)(confThreshold * 100);
    cv::createTrackbar("Limiar: %", nomeJanela, &initialConf, 99,
                       alterarLimiarConfianca);
  }

  // Carrega o arquivo ou a câmera de vídeo
  cv::VideoCapture videoCap;
  if (tipo.compare("2") == 0) {
    // Carrega arquivo
    videoCap.open(tipo);
  } else {
    // Carrega a primeira câmera encontrada
    for (int pos = 0; pos < 5; pos++) {
      if (videoCap.open(pos)) {
        break;
      }
      videoCap.release();
    }
  }

  cv::Mat frame;
  while (cv::waitKey(1) < 0) {
    videoCap >> frame;
    if (frame.empty()) {
      cv::waitKey(500);
    }

    if (!frame.empty()) {

      preprocess(frame, net, cv::Size(inpWidth, inpHeight), scale, mean,
                 swapRB);

      std::vector<cv::Mat> outs;
      net.forward(outs, outNames);

      postprocess(frame, outs, net, backendPreferencial);

      // Adiciona informações úteis
      std::vector<double> layersTimes;
      double freq = cv::getTickFrequency() / 1000;
      double t = net.getPerfProfile(layersTimes) / freq;
      std::string label = cv::format("Inference time: %.2f ms", t);
      cv::putText(frame, label, cv::Point(015, 15), cv::FONT_HERSHEY_SIMPLEX, 1,
                  cv::Scalar(0, 255, 0), 2);
    }
    cv::imshow(nomeJanela, frame);
  }

  return 0;
}

void exibirConfiguracoesAcessiveis() {
  std::vector<std::pair<cv::dnn::Backend, cv::dnn::Target>> availables =
      cv::dnn::getAvailableBackends();

  std::cout << "Tecnicas de processamento aceita:" << std::endl;
  for (auto &&available : availables) {
    std::cout << "Back: " << available.first << "Target: " << available.second
              << std::endl;
  }

  std::cout << "Tecnicas selecionada:" << std::endl;
  std::cout << "Back: " << backendPreferencial
            << "Target: " << targetPreferencial << std::endl
            << std::endl;
}

bool coletarParametros() {
  std::cout << "Informe a origem dos arquivos:" << std::endl
            << " 1: Webcam" << std::endl
            << " 2: Arquivo de vídeo" << std::endl;

  std::cin >> tipo;

  if (tipo.compare("2") == 0) {
    std::cout << "Informe o caminho do arquivo de vídeo:" << std::endl;
    try {
      tipo = cv::samples::findFile(tipo);
    } catch (cv::Exception ex) {
      std::cout << "Caminho inválido";
      return false;
    }

  } else if (tipo.compare("1") != 0) {
    std::cout << "Tipo inválido";
    return false;
  }

  if (!configCaminho.length()) {
    std::cout
        << "Informe o caminho do arquivo com as configurações da rede neural:"
        << std::endl;

    std::cin >> configCaminho;
    try {
      configCaminho = cv::samples::findFile(configCaminho);
    } catch (cv::Exception ex) {
      std::cout << "Caminho inválido";
      return false;
    }
  }
  if (!modelCaminho.length()) {
    std::cout << "Informe o caminho do arquivo com a rede neural treinada:"
              << std::endl;

    std::cin >> modelCaminho;
    try {
      modelCaminho = cv::samples::findFile(modelCaminho);
    } catch (cv::Exception ex) {
      std::cout << "Caminho inválido";
      return false;
    }
  }
  if (!labelCaminho.length()) {
    std::cout << "Informe o caminho do arquivo com os nomes dos rótulos, ou 0:"
              << std::endl;

    std::cin >> labelCaminho;
    if (labelCaminho.length()) {
      if (labelCaminho.compare("0") == 0) {
        labelCaminho = "";
      } else {
        try {
          labelCaminho = cv::samples::findFile(labelCaminho);
        } catch (cv::Exception ex) {
          std::cout << "Caminho inválido";
          return false;
        }
      }
    }
  }

  return true;
}

void carregarConfiguracoes() {

  if (labelCaminho.length()) {
    std::ifstream ifs(labelCaminho.c_str());
    if (!ifs.is_open()) {
      CV_Error(cv::Error::StsError,
               "Arquivo " + labelCaminho + " não encontrado");
    }
    std::string line;
    while (std::getline(ifs, line)) {
      rotulos.push_back(line);
    }
  }
}

void alterarLimiarConfianca(int pos, void *) { confThreshold = pos * 0.01f; }

inline void preprocess(const cv::Mat &frame, cv::dnn::Net &net,
                       cv::Size inpSize, float scale, const cv::Scalar &mean,
                       bool swapRB) {
  static cv::Mat blob;
  // Create a 4D blob from a frame.
  if (inpSize.width <= 0)
    inpSize.width = frame.cols;
  if (inpSize.height <= 0)
    inpSize.height = frame.rows;
  cv::dnn::blobFromImage(frame, blob, 1.0, inpSize, cv::Scalar(), swapRB, false,
                         CV_8U);

  // Run a model.
  net.setInput(blob, "", scale, mean);
  if (net.getLayer(0)->outputNameToIndex("im_info") !=
      -1) // Faster-RCNN or R-FCN
  {
    resize(frame, frame, inpSize);
    cv::Mat imInfo =
        (cv::Mat_<float>(1, 3) << inpSize.height, inpSize.width, 1.6f);
    net.setInput(imInfo, "im_info");
  }
}

void postprocess(cv::Mat &frame, const std::vector<cv::Mat> &outs,
                 cv::dnn::Net &net, int backend) {
  static std::vector<int> outLayers = net.getUnconnectedOutLayers();
  static std::string outLayerType = net.getLayer(outLayers[0])->type;

  std::vector<int> classIds;
  std::vector<float> confidences;
  std::vector<cv::Rect> boxes;
  if (outLayerType == "DetectionOutput") {
    // Network produces output blob with a shape 1x1xNx7 where N is a number of
    // detections and an every detection is a vector of values
    // [batchId, classId, confidence, left, top, right, bottom]
    CV_Assert(outs.size() > 0);
    for (size_t k = 0; k < outs.size(); k++) {
      float *data = (float *)outs[k].data;
      for (size_t i = 0; i < outs[k].total(); i += 7) {
        float confidence = data[i + 2];
        if (confidence > confThreshold) {
          int left = (int)data[i + 3];
          int top = (int)data[i + 4];
          int right = (int)data[i + 5];
          int bottom = (int)data[i + 6];
          int width = right - left + 1;
          int height = bottom - top + 1;
          if (width <= 2 || height <= 2) {
            left = (int)(data[i + 3] * frame.cols);
            top = (int)(data[i + 4] * frame.rows);
            right = (int)(data[i + 5] * frame.cols);
            bottom = (int)(data[i + 6] * frame.rows);
            width = right - left + 1;
            height = bottom - top + 1;
          }
          classIds.push_back((int)(data[i + 1]) -
                             1); // Skip 0th background class id.
          boxes.push_back(cv::Rect(left, top, width, height));
          confidences.push_back(confidence);
        }
      }
    }
  } else if (outLayerType == "Region") {
    for (size_t i = 0; i < outs.size(); ++i) {
      // Network produces output blob with a shape NxC where N is a number of
      // detected objects and C is a number of classes + 4 where the first 4
      // numbers are [center_x, center_y, width, height]
      float *data = (float *)outs[i].data;
      for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols) {
        cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
        cv::Point classIdPoint;
        double confidence;
        minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
        if (confidence > confThreshold) {
          int centerX = (int)(data[0] * frame.cols);
          int centerY = (int)(data[1] * frame.rows);
          int width = (int)(data[2] * frame.cols);
          int height = (int)(data[3] * frame.rows);
          int left = centerX - width / 2;
          int top = centerY - height / 2;

          classIds.push_back(classIdPoint.x);
          confidences.push_back((float)confidence);
          boxes.push_back(cv::Rect(left, top, width, height));
        }
      }
    }
  } else
    CV_Error(cv::Error::StsNotImplemented,
             "Unknown output layer type: " + outLayerType);

  // NMS is used inside Region layer only on DNN_BACKEND_OPENCV for another
  // backends we need NMS in sample or NMS is required if number of outputs > 1
  if (outLayers.size() > 1 ||
      (outLayerType == "Region" && backend != cv::dnn::DNN_BACKEND_OPENCV)) {
    std::map<int, std::vector<size_t>> class2indices;
    for (size_t i = 0; i < classIds.size(); i++) {
      if (confidences[i] >= confThreshold) {
        class2indices[classIds[i]].push_back(i);
      }
    }
    std::vector<cv::Rect> nmsBoxes;
    std::vector<float> nmsConfidences;
    std::vector<int> nmsClassIds;
    for (std::map<int, std::vector<size_t>>::iterator it =
             class2indices.begin();
         it != class2indices.end(); ++it) {
      std::vector<cv::Rect> localBoxes;
      std::vector<float> localConfidences;
      std::vector<size_t> classIndices = it->second;
      for (size_t i = 0; i < classIndices.size(); i++) {
        localBoxes.push_back(boxes[classIndices[i]]);
        localConfidences.push_back(confidences[classIndices[i]]);
      }
      std::vector<int> nmsIndices;
      cv::dnn::NMSBoxes(localBoxes, localConfidences, confThreshold,
                        nmsThreshold, nmsIndices);
      for (size_t i = 0; i < nmsIndices.size(); i++) {
        size_t idx = nmsIndices[i];
        nmsBoxes.push_back(localBoxes[idx]);
        nmsConfidences.push_back(localConfidences[idx]);
        nmsClassIds.push_back(it->first);
      }
    }
    boxes = nmsBoxes;
    classIds = nmsClassIds;
    confidences = nmsConfidences;
  }

  for (size_t idx = 0; idx < boxes.size(); ++idx) {
    cv::Rect box = boxes[idx];
    drawPred(classIds[idx], confidences[idx], box.x, box.y, box.x + box.width,
             box.y + box.height, frame);
  }
}

void drawPred(int classId, float conf, int left, int top, int right, int bottom,
              cv::Mat &frame) {
  rectangle(frame, cv::Point(left, top), cv::Point(right, bottom),
            cv::Scalar(0, 255, 0));

  std::string label = cv::format("%.2f", conf);
  if (!rotulos.empty()) {
    CV_Assert(classId < (int)rotulos.size());
    label = rotulos[classId] + ": " + label;
  }

  int baseLine;
  cv::Size labelSize =
      getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

  top = cv::max(top, labelSize.height);
  rectangle(frame, cv::Point(left, top - labelSize.height),
            cv::Point(left + labelSize.width, top + baseLine),
            cv::Scalar::all(255), cv::FILLED);
  putText(frame, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.5,
          cv::Scalar());
}
