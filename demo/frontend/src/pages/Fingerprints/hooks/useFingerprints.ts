import { useLayoutEffect } from "react";
import { useStores } from "stores/rootStore";

export const useFingerprints = () => {
  const { songsStore, functionalityStore } = useStores();
  useLayoutEffect(() => {
    functionalityStore.triggerActivity(true);
    songsStore.getTotalFingerprints().finally(() => {
      functionalityStore.triggerActivity(false);
    });
  }, []);
};
