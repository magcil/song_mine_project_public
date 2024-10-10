import { useLayoutEffect } from "react";
import balenaService from "services/balenaService";
import { useStores } from "stores/rootStore";

export const useDevices = () => {
  const { deviceStore, functionalityStore } = useStores();

  useLayoutEffect(() => {
    functionalityStore.triggerActivity(true);
    deviceStore.getDevices().finally(() => {
      functionalityStore.triggerActivity(false);
    });
    balenaService.getAllReleases().then((releases) => {
      // console.log(releases);
    });
  }, []);
};
