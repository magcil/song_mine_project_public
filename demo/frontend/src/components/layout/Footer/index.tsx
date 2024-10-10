import { Col, Layout, Row } from "antd";

function Footer() {
  const { Footer: AntFooter } = Layout;

  return (
    <AntFooter style={{ background: "#fafafa" }}>
      <Row className="just">
        <Col xs={24} md={12} lg={12}>
          <div className="copyright">
            Museek Dashboard ©{new Date().getFullYear()} Created by{" "}
            <a
              href="https://labs-repos.iit.demokritos.gr/MagCIL/"
              target="_blank"
              rel="noreferrer"
            >
              MagCIL
            </a>
          </div>
        </Col>
      </Row>
    </AntFooter>
  );
}

export default Footer;
