--
-- PostgreSQL database dump
--

-- Dumped from database version 14.7 (Ubuntu 14.7-0ubuntu0.22.04.1)
-- Dumped by pg_dump version 14.7 (Ubuntu 14.7-0ubuntu0.22.04.1)

-- Started on 2023-04-18 10:54:53 CST

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- TOC entry 321 (class 1259 OID 233530)
-- Name: stock_info; Type: TABLE; Schema: public; Owner: capstone
--

CREATE TABLE public.stock_info (
    last_update date DEFAULT CURRENT_DATE NOT NULL,
    symbol character varying(255) NOT NULL,
    info jsonb NOT NULL
);


ALTER TABLE public.stock_info OWNER TO capstone;

--
-- TOC entry 3295 (class 2606 OID 233574)
-- Name: stock_info stock_info_pkey; Type: CONSTRAINT; Schema: public; Owner: capstone
--

ALTER TABLE ONLY public.stock_info
    ADD CONSTRAINT stock_info_pkey PRIMARY KEY (last_update, symbol);


-- Completed on 2023-04-18 10:54:53 CST

--
-- PostgreSQL database dump complete
--

