(defproject clj-ml "0.0.0"
  :description "Clojure Machine Learning Library"
  :license {:name "MIT Open Source License"
            :url "https://opensource.org/licenses/MIT"}
  :dependencies [[org.clojure/clojure "1.8.0"]]
  :main ^:skip-aot clj-ml.core
  :target-path "target/%s"
  :profiles {:uberjar {:aot :all}})
