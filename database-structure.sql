-- phpMyAdmin SQL Dump
-- version 4.8.1
-- https://www.phpmyadmin.net/
--
-- Host: localhost
-- Generation Time: May 30, 2018 at 11:32 AM
-- Server version: 5.7.22
-- PHP Version: 7.2.5

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
SET AUTOCOMMIT = 0;
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `keywords_net`
--

-- --------------------------------------------------------

--
-- Table structure for table `analysissingle512_authors`
--

CREATE TABLE `analysissingle512_authors` (
  `author_id` int(11) NOT NULL,
  `first_paper_date` date NOT NULL,
  `broadness` double DEFAULT NULL,
  `broadness_lda` double DEFAULT NULL,
  `train` tinyint(1) DEFAULT NULL,
  `train_real` tinyint(1) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

-- --------------------------------------------------------

--
-- Table structure for table `analysissingle512_fast_citations`
--

CREATE TABLE `analysissingle512_fast_citations` (
  `cited_paper` int(11) NOT NULL,
  `cited_paper_date_created` date NOT NULL,
  `citing_paper` int(11) NOT NULL,
  `citing_paper_date_created` date NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

-- --------------------------------------------------------

--
-- Table structure for table `analysissingle512_fast_coauthors`
--

CREATE TABLE `analysissingle512_fast_coauthors` (
  `analysis_author_id` int(11) NOT NULL,
  `coauthor_id` int(11) NOT NULL,
  `coauthor_pagerank` float DEFAULT NULL,
  `first_date` date NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

-- --------------------------------------------------------

--
-- Table structure for table `analysissingle512_fast_paper_authors`
--

CREATE TABLE `analysissingle512_fast_paper_authors` (
  `paper_id` int(11) NOT NULL,
  `author_id` int(11) NOT NULL,
  `date_created` date NOT NULL,
  `pagerank` float DEFAULT NULL,
  `length` int(11) DEFAULT NULL,
  `jif` float NOT NULL,
  `published` tinyint(4) NOT NULL,
  `journal` varchar(255) DEFAULT NULL,
  `country` varchar(5) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

-- --------------------------------------------------------

--
-- Table structure for table `analysissingle512_hindex_data`
--

CREATE TABLE `analysissingle512_hindex_data` (
  `author_id` int(11) NOT NULL,
  `predict_after_years` int(11) NOT NULL,
  `hindex_before` int(11) NOT NULL,
  `hindex_after` int(11) NOT NULL,
  `hindex_cumulative` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

-- --------------------------------------------------------

--
-- Table structure for table `analysissingle512_nc_data`
--

CREATE TABLE `analysissingle512_nc_data` (
  `author_id` int(11) NOT NULL,
  `predict_after_years` int(11) NOT NULL,
  `nc_before` int(11) NOT NULL,
  `nc_after` int(11) NOT NULL,
  `nc_cumulative` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

-- --------------------------------------------------------

--
-- Table structure for table `authors`
--

CREATE TABLE `authors` (
  `id` int(11) NOT NULL,
  `name` varchar(255) NOT NULL,
  `broadness` double DEFAULT NULL,
  `broadness_lda` double DEFAULT NULL,
  `continuous_id` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

-- --------------------------------------------------------

--
-- Table structure for table `categories`
--

CREATE TABLE `categories` (
  `id` int(11) NOT NULL,
  `name` varchar(50) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

-- --------------------------------------------------------

--
-- Table structure for table `citations`
--

CREATE TABLE `citations` (
  `citing_paper` int(11) NOT NULL,
  `cited_paper` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

-- --------------------------------------------------------

--
-- Table structure for table `jif`
--

CREATE TABLE `jif` (
  `journal` varchar(255) NOT NULL,
  `issn` varchar(30) NOT NULL,
  `year` int(11) NOT NULL,
  `jif` float NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

-- --------------------------------------------------------

--
-- Table structure for table `papers`
--

CREATE TABLE `papers` (
  `id` int(11) NOT NULL,
  `arxiv_id` varchar(100) NOT NULL,
  `date_created` date NOT NULL,
  `date_updated` date DEFAULT NULL,
  `length` int(11) DEFAULT NULL,
  `journal` varchar(255) DEFAULT NULL,
  `num_authors` int(11) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

-- --------------------------------------------------------

--
-- Table structure for table `paper_authors`
--

CREATE TABLE `paper_authors` (
  `paper_id` int(11) NOT NULL,
  `author_id` int(11) NOT NULL,
  `country` varchar(5) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

-- --------------------------------------------------------

--
-- Table structure for table `paper_categories`
--

CREATE TABLE `paper_categories` (
  `paper_id` int(11) NOT NULL,
  `category_id` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

--
-- Indexes for dumped tables
--

--
-- Indexes for table `analysissingle512_authors`
--
ALTER TABLE `analysissingle512_authors`
  ADD UNIQUE KEY `author_id` (`author_id`);

--
-- Indexes for table `analysissingle512_fast_citations`
--
ALTER TABLE `analysissingle512_fast_citations`
  ADD KEY `cited_paper` (`cited_paper`),
  ADD KEY `citing_paper` (`citing_paper`);

--
-- Indexes for table `analysissingle512_fast_coauthors`
--
ALTER TABLE `analysissingle512_fast_coauthors`
  ADD UNIQUE KEY `analysis_author_id_2` (`analysis_author_id`,`coauthor_id`),
  ADD KEY `analysis_author_id` (`analysis_author_id`),
  ADD KEY `coauthor_id` (`coauthor_id`);

--
-- Indexes for table `analysissingle512_fast_paper_authors`
--
ALTER TABLE `analysissingle512_fast_paper_authors`
  ADD UNIQUE KEY `paper_id_2` (`paper_id`,`author_id`),
  ADD KEY `paper_id` (`paper_id`),
  ADD KEY `author_id` (`author_id`);

--
-- Indexes for table `analysissingle512_hindex_data`
--
ALTER TABLE `analysissingle512_hindex_data`
  ADD UNIQUE KEY `author_id` (`author_id`,`predict_after_years`) USING BTREE;

--
-- Indexes for table `analysissingle512_nc_data`
--
ALTER TABLE `analysissingle512_nc_data`
  ADD UNIQUE KEY `author_id` (`author_id`,`predict_after_years`) USING BTREE;

--
-- Indexes for table `authors`
--
ALTER TABLE `authors`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `name` (`name`),
  ADD UNIQUE KEY `continous_id` (`continuous_id`);

--
-- Indexes for table `categories`
--
ALTER TABLE `categories`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `name` (`name`);

--
-- Indexes for table `citations`
--
ALTER TABLE `citations`
  ADD KEY `cited_paper` (`cited_paper`),
  ADD KEY `citing_paper` (`citing_paper`);

--
-- Indexes for table `jif`
--
ALTER TABLE `jif`
  ADD UNIQUE KEY `journal` (`journal`,`year`);

--
-- Indexes for table `papers`
--
ALTER TABLE `papers`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `arxiv_id` (`arxiv_id`);

--
-- Indexes for table `paper_authors`
--
ALTER TABLE `paper_authors`
  ADD UNIQUE KEY `paper_id` (`paper_id`,`author_id`),
  ADD KEY `author_id` (`author_id`);

--
-- Indexes for table `paper_categories`
--
ALTER TABLE `paper_categories`
  ADD KEY `category_id` (`category_id`),
  ADD KEY `paper_id` (`paper_id`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `authors`
--
ALTER TABLE `authors`
  MODIFY `continuous_id` int(11) NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT for table `categories`
--
ALTER TABLE `categories`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT for table `papers`
--
ALTER TABLE `papers`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT;

--
-- Constraints for dumped tables
--

--
-- Constraints for table `citations`
--
ALTER TABLE `citations`
  ADD CONSTRAINT `citations_ibfk_1` FOREIGN KEY (`cited_paper`) REFERENCES `papers` (`id`) ON DELETE CASCADE ON UPDATE CASCADE,
  ADD CONSTRAINT `citations_ibfk_2` FOREIGN KEY (`citing_paper`) REFERENCES `papers` (`id`) ON DELETE CASCADE ON UPDATE CASCADE;

--
-- Constraints for table `paper_authors`
--
ALTER TABLE `paper_authors`
  ADD CONSTRAINT `paper_authors_ibfk_2` FOREIGN KEY (`paper_id`) REFERENCES `papers` (`id`) ON DELETE CASCADE ON UPDATE CASCADE,
  ADD CONSTRAINT `paper_authors_ibfk_3` FOREIGN KEY (`author_id`) REFERENCES `authors` (`id`);

--
-- Constraints for table `paper_categories`
--
ALTER TABLE `paper_categories`
  ADD CONSTRAINT `paper_categories_ibfk_1` FOREIGN KEY (`category_id`) REFERENCES `categories` (`id`) ON DELETE CASCADE ON UPDATE CASCADE,
  ADD CONSTRAINT `paper_categories_ibfk_2` FOREIGN KEY (`paper_id`) REFERENCES `papers` (`id`) ON DELETE CASCADE ON UPDATE CASCADE;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
