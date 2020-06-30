(ns clj-ml.core
  (:require [clj-ml.perceptron :as perceptron])
  (:gen-class))


(defn print-help
  []
  (println
    (str
      "\n"
      "USAGE: lein run <classifier>\n"
      "Supported classifiers:\n"
      "\tperceptron\n")))

(defn -main
  [& args]
  (case (first args)
    "perceptron"
      (perceptron/circle-test :act-fn (second args))
    (print-help)))
